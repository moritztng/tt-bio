#!/usr/bin/env python3
"""What the diffusion module's backward costs, in tokens/s.

The campaign cannot price a training step without this number and nobody had measured it.
The arms are the full 24-block token DiT at a production crop:

  fwd      the forward alone, nothing taped -- what inference runs today
  fwd+bwd  the same forward taped, plus the backward

and they are INTERLEAVED (fwd, fwd+bwd, fwd, ...), because a compile or a warm-up bias that
lands entirely on whichever arm ran first is the standard way to get a wrong A/B on this
stack. The A/A floor is the spread of the `fwd` repeats among themselves: a difference
inside it is not a difference.

Clock discipline: AICLK is sampled DURING each timed repeat by a thread, not before it, and
the per-arm minimum and median are reported. A number without a during-sampled clock is not
a measurement on Blackhole.

Memory: `ttnn` DRAM allocation is read at the peak of each arm, which is what the tape
retains -- the number `ptx-crop` needs, since fp32 doubles it per retained tensor.
"""
from __future__ import annotations

import argparse, json, os, statistics, sys, threading, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import dit

SYS = Path("/sys/class/tenstorrent")


def held_nodes():
    """The `/dev/tenstorrent/N` this process actually opened, read off its own fds.

    Not optional bookkeeping. `TT_VISIBLE_DEVICES` is a UMD index and the sysfs nodes are
    not the same numbering: on qb1, `TT_VISIBLE_DEVICES=2` opens `/dev/tenstorrent/3`. The
    first two passes of this measurement sampled `tenstorrent!2`, read 800 MHz flat through
    20 s of sustained load, and were one step from being written up as a governor artifact --
    while the card actually computing, node 3, held 1350 MHz in every repeat. A clock keyed
    to the launch flag can watch a different chip than the one doing the work.
    """
    import glob
    return sorted({int(os.readlink(f).rsplit("/", 1)[1])
                  for f in glob.glob("/proc/self/fd/*")
                  if os.path.islink(f) and os.readlink(f).startswith("/dev/tenstorrent/")})


class ClockWatch(threading.Thread):
    """Samples every card's AICLK while a timed region runs. Opens no device."""

    def __init__(self, period=0.01):
        super().__init__(daemon=True)
        self.period, self.stop_flag, self.samples = period, threading.Event(), []

    def run(self):
        while not self.stop_flag.is_set():
            row = {}
            for p in sorted(SYS.glob("tenstorrent!*")):
                try:
                    row[p.name.split("!")[1]] = int((p / "tt_aiclk").read_text().strip())
                except (OSError, ValueError):
                    pass
            self.samples.append(row)
            time.sleep(self.period)

    def report(self):
        out = {}
        for k in (self.samples[0] if self.samples else {}):
            v = [s[k] for s in self.samples if k in s]
            out[k] = {"min": min(v), "median": int(statistics.median(v)), "max": max(v),
                      "n": len(v)}
        return out


def stack(ag, t, nt, blocks, *, cfg, bwcfg):
    """`blocks` DiT blocks chained, weights shared across blocks.

    Sharing the weights is deliberate and it is the honest choice for a THROUGHPUT arm: 24
    private copies would measure DRAM capacity, and the compute, the activation tape and the
    reduction lengths are identical either way. What it does change is the weight gradient's
    fan-in, which accumulates 24 times into one buffer instead of once into each of 24 --
    the same number of adds, so the backward's work is unchanged.
    """
    a = t["a"]
    for _ in range(blocks):
        t = dict(t, a=a)
        a = dit.block_tape(ag, t, nt, cfg=cfg, bwcfg=bwcfg, attn_cfg=None)
    return a


def dram(ttnn, dev):
    """Bytes of device DRAM in use, `tenstorrent.py:dram_peak`'s own arithmetic.

    NEVER call this inside a timed region. `ttnn.get_memory_view` behaves like a pipeline
    drain: a 117 aa fold measured 12.0 s with it off and 28.8 s with it on, so a timed run
    carrying this probe measures the probe. The memory pass runs untimed for that reason.
    """
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    return int(mv.total_bytes_per_bank - mv.total_bytes_free_per_bank) * int(mv.num_banks)


def one(ag, ttnn, tt, t, nt, blocks, cfg, bwcfg, seed_t, backward, probe=False):
    """One arm's work. `probe` reads DRAM at the tape's high-water mark and again after the
    backward, and is only ever set on the untimed memory pass."""
    dev_h = tt.get_device()
    peak = None
    if backward:
        out = stack(ag, t, nt, blocks, cfg=cfg, bwcfg=bwcfg)
        if probe:
            # Everything the forward retained, nothing freed yet -- the number `ptx-crop`
            # wants. Read again after, because a backward frees as it goes and the true
            # peak can be on either side.
            peak = dram(ttnn, dev_h)
        out.backward(seed=seed_t)
        if probe:
            peak = max(peak, dram(ttnn, dev_h))
    else:
        with ag.no_grad():
            out = stack(ag, t, nt, blocks, cfg=cfg, bwcfg=bwcfg)
        if probe:
            peak = dram(ttnn, dev_h)
    ttnn.synchronize_device(dev_h)
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=384, help="Protenix's training crop")
    ap.add_argument("--blocks", type=int, default=dit.BLOCKS)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--burn", type=float, default=15.0,
                    help="seconds of sustained load before timing, so the AICLK governor is "
                         "ramped; 0 reproduces the first pass, which read 800 MHz")
    ap.add_argument("--memory", action="store_true",
                    help="untimed DRAM pass. Never combined with the timed one: "
                         "ttnn.get_memory_view drains the pipeline.")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    dev_h = tt.get_device()

    torch_dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tt_dt = ttnn.float32 if args.dtype == "float32" else ttnn.bfloat16
    precise = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    cfg = precise if args.dtype == "float32" else ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=True)

    rng = np.random.default_rng(args.seed)
    raw = dit.params(rng, args.nt)

    def up(v):
        return ttnn.from_torch(torch.from_numpy(v).to(torch_dt), dtype=tt_dt,
                               layout=ttnn.TILE_LAYOUT, device=dev_h)

    base_dram = dram(ttnn, dev_h)
    seed_t = up(rng.standard_normal((args.nt, dit.C)))

    # Uploaded ONCE. Re-uploading per repeat put ~0.4 s of host staging between consecutive
    # device bursts, which is long enough for the AICLK governor to decay -- the first run of
    # this measured 44 ms bursts against a card reading 800 MHz throughout. Only the tape
    # wrappers are rebuilt per repeat; the device buffers are the same ones, which is also
    # what makes the timing the block's and not `from_torch`'s.
    resident = {k: up(v) for k, v in raw.items()}

    def fresh(requires_grad):
        return {k: ag.Tensor(v, requires_grad=(requires_grad and k in dit.WEIGHTS))
                for k, v in resident.items()}

    # warm-up: both arms compile before either is timed, so neither pays the other's
    # first-call cost.
    for bw in (False, True):
        one(ag, ttnn, tt, fresh(bw), args.nt, args.blocks, cfg, precise, seed_t, bw)

    # Governor burn-in. On Blackhole the clock sets the time, and the governor ramps on
    # sustained load, so a measurement made of short bursts reads a decayed clock and prices
    # the block against a card that a real training loop would never present. Run the heavier
    # arm back to back until the card has been busy for `--burn` seconds, then time.
    if args.burn > 0:
        t_end = time.perf_counter() + args.burn
        while time.perf_counter() < t_end:
            one(ag, ttnn, tt, fresh(True), args.nt, args.blocks, cfg, precise, seed_t, True)
        print(f"burn-in {args.burn}s done, clocks now "
              f"{ {k: int((Path('/sys/class/tenstorrent') / f'tenstorrent!{k}' / 'tt_aiclk').read_text()) for k in ('0','1','2','3') if (Path('/sys/class/tenstorrent') / f'tenstorrent!{k}').exists()} }",
              flush=True)

    rows = {"fwd": [], "fwd_bwd": []}
    peaks = {"fwd": [], "fwd_bwd": []}
    clocks = {"fwd": [], "fwd_bwd": []}
    for r in range(args.reps):
        for arm, bw in (("fwd", False), ("fwd_bwd", True)):
            t = fresh(bw)
            ttnn.synchronize_device(dev_h)
            w = ClockWatch(); w.start()
            t0 = time.perf_counter()
            peak = one(ag, ttnn, tt, t, args.nt, args.blocks, cfg, precise, seed_t, bw)
            dt = time.perf_counter() - t0
            w.stop_flag.set(); w.join()
            rows[arm].append(dt); peaks[arm].append(peak); clocks[arm].append(w.report())
            print(f"rep {r} {arm:8s} {dt*1000:8.1f} ms  {args.nt/dt:8.1f} tok/s  "
                  f"peak {(peak or 0)/1e9:.3f} GB  clk {w.report()}", flush=True)

    if args.memory:
        mem = {}
        for arm, bw in (("fwd", False), ("fwd_bwd", True)):
            mem[arm] = one(ag, ttnn, tt, fresh(bw), args.nt, args.blocks, cfg, precise,
                           seed_t, bw, probe=True)
            print(f"{arm:8s} DRAM {mem[arm]/2**30:.3f} GiB", flush=True)
        rep = {"nt": args.nt, "blocks": args.blocks, "dtype": args.dtype,
               "device_nodes_held": held_nodes(),
               "dram_base_bytes": base_dram, "dram_bytes": mem,
               "retained_bytes": mem["fwd_bwd"] - mem["fwd"],
               "retained_bytes_per_block": (mem["fwd_bwd"] - mem["fwd"]) // args.blocks}
        print(json.dumps(rep, indent=1))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rep, indent=1))
        return 0

    def summarise(arm):
        s = sorted(rows[arm])
        med = statistics.median(s)
        return {"seconds_median": round(med, 4), "seconds_min": round(s[0], 4),
                "seconds_max": round(s[-1], 4),
                "spread_pct": round(100 * (s[-1] - s[0]) / med, 2),
                "tokens_per_s": round(args.nt / med, 2),
                "dram_peak_bytes": (max(p for p in peaks[arm] if p is not None)
                                    if any(p is not None for p in peaks[arm]) else None),
                "aiclk_during": clocks[arm]}

    f, b = summarise("fwd"), summarise("fwd_bwd")
    nodes = held_nodes()
    rep = {"nt": args.nt, "blocks": args.blocks, "dtype": args.dtype, "reps": args.reps,
           "seed": args.seed, "interleaved": True,
           "device_nodes_held": nodes,
           "aiclk_of_computing_node": {
               str(n): {"min": min(c[str(n)]["min"] for c in clocks["fwd_bwd"] if str(n) in c),
                        "max": max(c[str(n)]["max"] for c in clocks["fwd_bwd"] if str(n) in c)}
               for n in nodes if any(str(n) in c for c in clocks["fwd_bwd"])},
           "dram_base_bytes": base_dram,
           "arms": {"fwd": f, "fwd_bwd": b},
           "aa_floor_pct": f["spread_pct"],
           "backward_multiple": round(b["seconds_median"] / f["seconds_median"], 3),
           "tokens_per_s_cost_pct": round(100 * (1 - b["tokens_per_s"] / f["tokens_per_s"]), 2),
           "retained_bytes": (b["dram_peak_bytes"] - f["dram_peak_bytes"])
                             if None not in (b["dram_peak_bytes"], f["dram_peak_bytes"]) else None}
    print(json.dumps({k: v for k, v in rep.items() if k != "arms"}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
