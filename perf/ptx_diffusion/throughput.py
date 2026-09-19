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

import argparse, json, statistics, sys, threading, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import dit

SYS = Path("/sys/class/tenstorrent")


class ClockWatch(threading.Thread):
    """Samples every card's AICLK while a timed region runs. Opens no device."""

    def __init__(self, period=0.05):
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
    try:
        return int(dev.allocated_bytes_per_bank(ttnn.BufferType.DRAM)) * dev.num_dram_channels()
    except Exception:
        try:
            return int(ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).total_allocated_bytes)
        except Exception:
            return None


def one(ag, ttnn, tt, t, nt, blocks, cfg, bwcfg, seed_t, backward):
    dev_h = tt.get_device()
    if backward:
        out = stack(ag, t, nt, blocks, cfg=cfg, bwcfg=bwcfg)
        # Read at the tape's high-water mark (everything the forward retained, nothing freed
        # yet) and again after the backward has run, and keep the larger. Reading only the
        # first would quote what the tape RETAINS and call it the peak; reading only the
        # second would miss it, because a backward frees as it goes.
        peak = dram(ttnn, dev_h)
        out.backward(seed=seed_t)
        after = dram(ttnn, dev_h)
        if None not in (peak, after):
            peak = max(peak, after)
    else:
        with ag.no_grad():
            out = stack(ag, t, nt, blocks, cfg=cfg, bwcfg=bwcfg)
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

    def fresh(requires_grad):
        return {k: ag.Tensor(up(v), requires_grad=(requires_grad and k in dit.WEIGHTS))
                for k, v in raw.items()}

    # warm-up: both arms compile before either is timed, so neither pays the other's
    # first-call cost.
    for bw in (False, True):
        one(ag, ttnn, tt, fresh(bw), args.nt, args.blocks, cfg, precise, seed_t, bw)

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
            for k, v in t.items():
                ttnn.deallocate(v.value)

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
    rep = {"nt": args.nt, "blocks": args.blocks, "dtype": args.dtype, "reps": args.reps,
           "seed": args.seed, "interleaved": True,
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
