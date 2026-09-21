#!/usr/bin/env python3
"""Price the lever's two halves on the chip: the split it deletes, the matmul it moves.

`c12-diffusion-head-major` pass 10 timed whole chains (A0 = linear+split, A1 = minimal_matmul+split,
B = head-major) and got a number at ONE signature, `apb_trunk`, which is 1.85 % of the class. The
two signatures carrying the other 94 % returned nothing at all, because arm B hangs the device
there. So the book's 0.2340 s still rests on a signature that is almost none of it.

The lever is a difference of two terms and BOTH are measurable without arm B:

    SPLIT   nlp_create_qkv_heads alone     what the head-major writer deletes -- the whole prize
    LIN     ttnn.linear alone              the projection that ships today
    MM      wheel minimal_matmul alone     the op class pass 2's reprice moves the matmul to
    GEN     generic_op transcription alone the kernel arm B actually runs, plain writer

    delivered_per_call  =  SPLIT - (B_matmul - LIN),  and  B_matmul >= GEN,

because the head-major writer does the plain writer's work plus a different destination address.
GEN is therefore a LOWER bound on the lever's cost and SPLIT is an UPPER bound on its saving: if
SPLIT - (GEN - LIN) is negative at a signature, the lever is refuted there without running arm B
at all. That is the reading pass 10 could not get.

One process per stage, every arm interleaved rep by rep inside it, cold rep discarded per arm,
A/A control in the same arm list, AICLK forced and sampled at 500 Hz in a subprocess DURING the
measurement.

    python3 time_arms.py --sig apb_trunk --arms LIN,MM,GEN,SPLIT,B,AA
    python3 time_arms.py --sig dit_token --arms LIN,MM,SPLIT,AA        # B hangs, GEN is stage C
"""
import argparse
import json
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
T0 = time.time()

# name -> batch, seq, c_in, heads, head_dim, padded_head_dim, site, split_calls_per_fold, in_situ_s
# The last two are `c12-tail-classes-screen`'s in-situ pricing of the SPLIT at 512 aa, 1350 MHz,
# which is the term this lever deletes. Reused from perf/c12_diffusion_head/ab_qkv.py's SIGS.
SIGS = {
    "apb_trunk": (1, 512, 384, 16, 32, 32, "apb", 264, 0.00581),
    "dit_token": (1, 512, 768, 16, 48, 64, "apb", 4800, 0.20470),
}


def say(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def build(device, sig_name):
    """Every arm at one signature, primitives as well as chains, from one set of operands."""
    import torch
    import ttnn
    from tt_bio import tenstorrent as TT
    from tt_bio import triatt_qkv as TQ
    from tt_bio import mm_generic as MG

    b, s, c_in, heads, hd, phd, site, _calls, _in_situ = SIGS[sig_name]
    torch.manual_seed(0)
    x_t = torch.randn(b, s, c_in, dtype=torch.bfloat16)
    w_t = (torch.randn(c_in, 3 * heads * phd, dtype=torch.bfloat16) * 0.02).to(torch.bfloat16)
    bias_t = (torch.randn(3 * heads * phd, dtype=torch.bfloat16) * 0.02).to(torch.bfloat16)
    mk = dict(layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16,
              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x, w, bias = (ttnn.from_torch(t, **mk) for t in (x_t, w_t, bias_t))

    cls = (ttnn.types.WormholeComputeKernelConfig if device.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    ckc = cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    cfg = TT._qkv_mm_config(x, w, site)
    assert cfg is not None, "no block config"
    blk = TT._mm_block_at_site((c_in + 31) // 32, (3 * heads * phd + 31) // 32, site)
    say(f"{sig_name}: site={site} block={blk} M={b * s} N={3 * heads * phd} "
        f"transpose_core_grid={b * s > 3 * heads * phd}")

    # The split's input is a projection output, made ONCE so the SPLIT arm times the split and
    # nothing else. Deallocating it inside the timing loop would make the second rep time a
    # different op, so the loop deallocates only what an arm returns.
    fixed = ttnn.linear(x, w, bias=bias, compute_kernel_config=ckc, core_grid=TT.CORE_GRID_MAIN)

    def split(full):
        return list(ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(full, 1), num_heads=heads, num_kv_heads=heads, transpose_k_heads=False))

    def lin():
        return [ttnn.linear(x, w, bias=bias, compute_kernel_config=ckc,
                            core_grid=TT.CORE_GRID_MAIN)]

    def mm():
        return [ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, bias_tensor=bias, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg)]

    def gen():
        full = ttnn.allocate_tensor_on_device(
            ttnn.Shape([b, s, 3 * heads * phd]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
            device, ttnn.DRAM_MEMORY_CONFIG)
        MG.generic_minimal_matmul(device, x, w, [full], (blk, tuple(TT.COMPUTE_GRID_MAIN)),
                                  MG.ckc_args(ckc), {}, TQ.KERNEL_DIR, bias=bias)
        return [full]

    def split_only():
        return split(fixed)

    def a0():
        return split(ttnn.linear(x, w, bias=bias, compute_kernel_config=ckc,
                                 core_grid=TT.CORE_GRID_MAIN))

    def arm_b():
        TQ._APB_ENABLED = True
        o = TQ.qkv_heads(x, w, ckc, heads, phd, ttnn.bfloat16, cfg, bias=bias,
                         allow_m_le_n=True, site=site)
        assert o is not None, f"declined: {TQ.APB_REJECTS} {TQ.REJECTS}"
        return list(o)

    return {"LIN": lin, "MM": mm, "GEN": gen, "SPLIT": split_only,
            "A0": a0, "B": arm_b, "AA": a0}


def timed(device, arms, order, reps):
    """Interleaved rep by rep, cold rep discarded per arm, median of the rest."""
    import ttnn
    times = {k: [] for k in order}
    for rep in range(reps + 1):
        for key in order:
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            outs = arms[key]()
            ttnn.synchronize_device(device)
            dt = (time.perf_counter() - t0) * 1e3
            for o in outs:
                ttnn.deallocate(o)
            if rep:
                times[key].append(dt)
        if rep == 0:
            say("cold rep discarded")
    return {k: sorted(v)[len(v) // 2] for k, v in times.items()}, times


# What the device run RECORDS. Everything else in an artifact is derived, so a re-derive
# drops it first: merging a new derivation over an old one leaves both namings in the
# file and the reader cannot tell which arithmetic produced which key.
RECORDED = {"sig", "reps", "order", "ms_per_call", "raw_ms", "clock_during",
            "clock_forced_MHz", "clock_floor_MHz", "clock_admissible", "window_s",
            "split_calls_per_fold", "split_in_situ_s", "nodes"}


def derive(med, calls, in_situ):
    """Per-fold arithmetic, with the chain difference as the number of record.

    A standalone SPLIT arm is timed with its own `synchronize_device` on both sides, so it carries
    a dispatch latency the split does not pay inside the chain: 0.03752 ms standalone against
    0.02560 ms as A0 - LIN at `apb_trunk`. The chain difference is what the lever would actually
    recover, and it is the one that reproduces the in-situ price (1.014x at `dit_token`), so
    everything per-fold below is derived from it and the standalone arm is reported beside it as
    the upper bound it is.
    """
    d = {}
    if "A0" in med and "AA" in med:
        d["aa_floor_pct"] = abs(med["AA"] - med["A0"]) / med["A0"] * 100.0
    if "SPLIT" in med:
        d["split_standalone_ms_per_call"] = med["SPLIT"]
    if "A0" in med and "LIN" in med:
        split = med["A0"] - med["LIN"]
        d["split_in_chain_ms_per_call"] = split
        d["split_in_chain_s_per_fold"] = split * calls / 1e3
        d["split_vs_in_situ"] = (split * calls / 1e3) / in_situ if in_situ else None
        if "MM" in med:
            d["wheel_mm_move_ms_per_call"] = med["MM"] - med["LIN"]
            d["wheel_mm_move_s_per_fold"] = (med["MM"] - med["LIN"]) * calls / 1e3
            d["ceiling_if_writer_were_wheel_mm_s_per_fold"] = (
                (split - (med["MM"] - med["LIN"])) * calls / 1e3)
        if "GEN" in med:
            d["gen_over_lin"] = med["GEN"] / med["LIN"]
            d["transcription_move_ms_per_call"] = med["GEN"] - med["LIN"]
            d["lever_upper_bound_ms_per_call"] = split - (med["GEN"] - med["LIN"])
            d["lever_upper_bound_s_per_fold"] = (split - (med["GEN"] - med["LIN"])) * calls / 1e3
        if "B" in med:
            d["lever_measured_ms_per_call"] = med["A0"] - med["B"]
            d["lever_measured_s_per_fold"] = (med["A0"] - med["B"]) * calls / 1e3
            if "GEN" in med:
                d["head_major_writer_extra_ms_per_call"] = med["B"] - med["GEN"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", default="apb_trunk", choices=sorted(SIGS))
    ap.add_argument("--arms", default="LIN,MM,GEN,SPLIT,AA")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--floor", type=int, default=1200)
    ap.add_argument("--out", default="")
    ap.add_argument("--from", dest="src", default="",
                    help="re-derive per-fold arithmetic from a stored artifact, no device")
    a = ap.parse_args()
    order = [k.strip() for k in a.arms.split(",") if k.strip()]
    if a.src:
        rec = json.loads((OUT / a.src).read_text())
        rec = {k: v for k, v in rec.items() if k in RECORDED}
        _b, _s, _c, _h, _hd, _phd, _site, calls, in_situ = SIGS[rec["sig"]]
        rec.update(derive(rec["ms_per_call"], calls, in_situ))
        (OUT / a.src).write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "raw_ms"}, indent=1))
        return 0

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "perf" / "c12_compose"))
    import clk
    import ttnn                                                     # noqa: F401
    from tt_bio import tenstorrent as TT

    say(f"{a.sig}: opening device via tt_bio.tenstorrent.get_device()")
    dev = TT.get_device()
    nodes = clk.nodes_open_by_this_process()
    say(f"device open, arch={dev.arch()}, nodes {nodes}")
    held = clk.force(a.mhz, nodes)
    say(f"AICLK forced to {a.mhz} MHz on {held}")

    arms = build(dev, a.sig)
    missing = [k for k in order if k not in arms]
    if missing:
        raise SystemExit(f"unknown arms {missing}; have {sorted(arms)}")
    say(f"arms built, timing {order} x {a.reps} reps interleaved")

    sampler = clk.Sampler(nodes[0])
    t_start = time.time()
    med, raw = timed(dev, arms, order, a.reps)
    t_end = time.time()
    clock = sampler.stop()
    say(f"AICLK during the window: {clock}")

    b, s, c_in, heads, hd, phd, site, calls, in_situ = SIGS[a.sig]
    rec = {"sig": a.sig, "reps": a.reps, "order": order, "ms_per_call": med, "raw_ms": raw,
           "clock_during": clock, "clock_forced_MHz": a.mhz, "clock_floor_MHz": a.floor,
           "clock_admissible": bool(clock["n"] and clock["min"] >= a.floor),
           "window_s": round(t_end - t_start, 2), "split_calls_per_fold": calls,
           "split_in_situ_s": in_situ, "nodes": nodes}
    rec.update(derive(med, calls, in_situ))
    path = OUT / (a.out or f"time_{a.sig}.json")
    path.write_text(json.dumps(rec, indent=1) + "\n")
    for k in order:
        say(f"  {k:<6} {med[k]:.5f} ms/call")
    say(f"WROTE {path}")
    return 0 if rec["clock_admissible"] else 3


if __name__ == "__main__":
    sys.exit(main())
