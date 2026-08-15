#!/usr/bin/env python3
"""S-D: price the DRAM-write half of the trimul in-projection, and screen a dual-NOC drain.

The in-projection `[512,512,256] x [256,1024]` reads 128 MiB and writes 512 MiB, measures
2.722 ms/call against a 1.562 ms byte floor, and sits at 52 % of the compute roof and 57 % of the
byte roof at the same time (state/esmfold2-to-4x-per-dollar.md 4.0/4.1). Those two add to 109 %,
which is the additive `time = compute + write` signature rather than `max(...)`. 1084 calls per
fold, so the headroom is ~1.26 s.

Four arms, one process, alternating, batched timing (8 back-to-back calls, ONE synchronize at the
end, median of 5 batches). Never per-op-synced: the same tape read the same op at 28.9 ms and
2.2 ms in two arms, a 13x instrument artefact (memory
`tt-bio-isolated-op-timing-oversync-inflates-cost`).

  native    ttnn.experimental.minimal_matmul, the shipped call
  generic   tt_bio/mm_generic.py's transcription, unmodified. The transcription check.
  nowrite   same, the output CB drained and popped but the DRAM write not issued. Prices the
            write half. Output is garbage by construction, so this arm is a stopwatch.
  dualnoc   same, every second output tile issued on the other NOC. The writer RISC carries
            512 MiB on NOC_1 while the reader RISC carries 128 MiB on NOC_0, so half the drain
            moves onto an idle wire. Same tiles, same addresses, same bytes: bit-exact or broken,
            nothing in between.

PREDICTIONS, WRITTEN BEFORE THE RUN:
  generic reproduces native within 5 % and torch.equal.
  write half (generic - nowrite) is 0.9-1.3 ms of the 2.722 ms.
  dualnoc recovers >= 1.25x of the write half, 0.35-0.50 ms/call = 0.38-0.54 s of fold.

KILL GATES, all pre-committed:
  1. generic not torch.equal against native, or slower than 1.10x: the generic_op route is dead
     for this op. Stop and report.
  2. write half below 0.35 ms/call: the serialisation diagnosis is wrong for this shape, D is
     worth < 0.4 s, NO-GO.
  3. dualnoc below 1.25x on the write half: NO-GO, record it and move on.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import mm_generic as G

assert Path(T.__file__).resolve().is_relative_to(REPO), "tt_bio from %s" % T.__file__

SPLIT_KERNELS = REPO / "tt_bio" / "kernels" / "mm_split"
CALLS_PER_FOLD = 1084


def timed(fn, dev, reps=8, batches=5, warm=2):
    """Batched wall per call. One synchronize per batch, never one per call."""
    for _ in range(warm):
        for _ in range(reps):
            fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / reps)
    return st.median(ts) * 1e3, (max(ts) - min(ts)) / st.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--nout", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--arms", default="generic,nowrite,dualnoc")
    ap.add_argument("--out", default="perf/esmbeat/s_d_writesplit.json")
    args = ap.parse_args()
    S, K, N = args.n, args.k, args.nout

    dev = T.get_device()
    # TorchWrapper's own config (tt_bio/tenstorrent.py), which is what the fold runs.
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    gckc = (ttnn.MathFidelity.HiFi4, False, True, False)

    res = {"predictions": __doc__, "arms": [], "meta": {
        "shape": [S, S, K, N], "grid": list(T.COMPUTE_GRID_MAIN),
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "loadavg": os.getloadavg(),
        "ttnn": ttnn.__version__ if hasattr(ttnn, "__version__") else None,
        "calls_per_fold": CALLS_PER_FOLD}}

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    x = dram(torch.randn(S, S, K).to(torch.bfloat16))
    w = dram(torch.randn(K, N).to(torch.bfloat16))

    # The fold's in-projection (`_in_proj_rows` / `__call__` in tenstorrent.py) calls
    # `minimal_matmul` with NO `config=`, so its blocking is the op's own
    # `determine_default_block_sizes`, which returns (8, 8, 8) with subblocks (2, 2) under
    # fp32_dest_acc_en. `_MM_DEFAULT` is that tuple, and an explicit config equal to it is
    # byte-identical to `config=None` by construction, measured at max_abs 0.0
    # (tenstorrent.py, the _MM_DEFAULT comment). There is no (8, 32) entry in `_MM_BLOCK`.
    cfg = T._qkv_mm_config(x, w)
    blk = T._mm_block_for(w) or T._MM_DEFAULT
    gcfg = (blk, tuple(T.COMPUTE_GRID_MAIN))
    res["meta"]["mm_config"] = str(cfg)
    res["meta"]["block"] = list(blk)
    res["meta"]["native_unconfigured"] = cfg is None
    print(json.dumps(res["meta"]), flush=True)

    def native():
        if cfg is None:
            return ttnn.experimental.minimal_matmul(
                x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                compute_kernel_config=ckc)
        return ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg)

    ref = ttnn.to_torch(native())

    outs = {}
    want = [a for a in args.arms.split(",") if a]
    # DM_DYNAMIC_NOC on the dualnoc arm is not a tuning choice. Under the default
    # DM_DEDICATED_NOC the firmware only runs noc_local_state_init for the kernel's own NOC, so
    # the alt-NOC write never issues and the barrier spins: MEASURED as a device hang on the
    # first call, card 0, which cost a tt-smi reset.
    arms = [a for a in (("generic", {}, None, None),
                        ("nowrite", {"MM_NOWRITE": 1}, SPLIT_KERNELS, None),
                        ("dualnoc", {"MM_DUAL_NOC": 1}, SPLIT_KERNELS,
                         ttnn.NOC_MODE.DM_DYNAMIC_NOC))
            if a[0] in want]
    for name, *_ in arms:
        outs[name] = ttnn.allocate_tensor_on_device(
            ttnn.Shape([S, S, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
            ttnn.DRAM_MEMORY_CONFIG)
        # zero it, so an arm that writes nothing cannot inherit a correct answer
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.zeros(S, S, N, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            outs[name])

    def mk(name, defines, kdir, nmode):
        o = outs[name]
        return lambda: G.generic_minimal_matmul(
            dev, x, w, o, gcfg, gckc, defines, kdir, None, nmode)

    fns = {"native": native}
    for name, defines, kdir, nmode in arms:
        fns[name] = mk(name, defines, kdir, nmode)

    # correctness first, once per arm, before any timing
    for name, *_ in arms:
        t0 = time.perf_counter()
        fns[name]()
        build_s = time.perf_counter() - t0
        got = ttnn.to_torch(outs[name])
        eq = bool(torch.equal(got, ref))
        row = {"arm": name, "torch_equal": eq, "first_call_s": build_s}
        if not eq:
            d = (got.float() - ref.float()).abs()
            row["max_abs_err"] = float(d.max())
            row["nonzero_frac"] = float((got != 0).float().mean())
        res.setdefault("parity", []).append(row)
        print(json.dumps(row), flush=True)
        del got

    order = ["native"] + [a[0] for a in arms]
    for rnd in range(args.rounds):
        for name in order:
            ms, spread = timed(fns[name], dev)
            row = {"arm": name, "round": rnd, "ms": ms, "batch_spread": spread}
            res["arms"].append(row)
            print(json.dumps(row), flush=True)

    ms_a, _ = timed(fns["generic"], dev)
    ms_b, _ = timed(fns["generic"], dev)
    res["aa_pair_generic_ms"] = [ms_a, ms_b]

    med = {n: st.median([r["ms"] for r in res["arms"] if r["arm"] == n]) for n in order}
    write_half = med["generic"] - med.get("nowrite", med["generic"])
    dual_gain = med["generic"] - med.get("dualnoc", med["generic"])
    res["summary"] = {
        "median_ms": med,
        "aa_pair_generic_ms": res["aa_pair_generic_ms"],
        "aa_floor_ms": abs(ms_a - ms_b),
        "generic_vs_native": med["generic"] / med["native"],
        "write_half_ms": write_half,
        "dualnoc_saving_ms": dual_gain,
        "dualnoc_x_on_write_half": (write_half / (write_half - dual_gain))
                                   if write_half - dual_gain > 0 else None,
        "fold_s_if_shipped": dual_gain * CALLS_PER_FOLD / 1e3,
        "gate1_transcription_ok": (all(r["torch_equal"] for r in res["parity"]
                                       if r["arm"] == "generic")
                                   and med["generic"] / med["native"] <= 1.10),
        "gate2_write_half_ok": write_half >= 0.35,
        "gate3_dualnoc_ok": (dual_gain > 0 and write_half > 0
                             and (write_half / max(write_half - dual_gain, 1e-9)) >= 1.25),
        "dualnoc_bit_exact": all(r["torch_equal"] for r in res["parity"]
                                 if r["arm"] == "dualnoc"),
    }
    print(json.dumps(res["summary"], indent=1), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
