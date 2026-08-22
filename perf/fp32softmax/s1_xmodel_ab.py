#!/usr/bin/env python3
"""Interleaved fold A/B of the fp32-softmax height-shard plan (S1), on any pair-track model.

S1 lets `_fp32_softmax_l1_plan` choose the shard's core count where the tuned 8x8 rectangle cannot
divide any affordable block. It is a memory-config change only, so the fold is expected to be
bit-exact: same structure digest, same plDDT, and the only thing that moves is where the fp32 score
copy lives.

Why this harness and not one process per arm: the flag is flipped IN PROCESS
(`tenstorrent._FP32_SOFTMAX_L1_ANY_CORES` plus the plan cache and the refusal memo), so both arms
share one loaded checkpoint and one warm device, and the arms interleave inside one sweep. That is
the A/B the perf method asks for; `--arms AA` runs the same instrument as an A/A control.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python3 perf/fp32softmax/s1_xmodel_ab.py \
        --model rf3 --aa 704 --arms ABABAB --out /tmp/s1_rf3_704.json

Every fold's `fp32_softmax` counters are in the report, so an arm that never reached the lever is
visible instead of being read as "no effect" (the K6 lesson: assert the lever served before reading
a ratio).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def digest_dir(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--aa", type=int, default=704)
    ap.add_argument("--arms", default="ABABAB",
                    help="fold order, A = shipped 8x8 plan, B = S1. One cold pair is discarded.")
    ap.add_argument("--recycling_steps", type=int, default=1)
    ap.add_argument("--sampling_steps", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)

    from perf_regression import SPECS, _build_cfg
    sys.path.insert(0, str(ROOT / "perf" / "rf3"))
    from make_inputs import cdk2

    from tt_bio import tenstorrent as tt
    from tt_bio import esmfold2 as _E
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts

    _E.set_progress(lambda *a, **k: None)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    work = Path(tempfile.mkdtemp(prefix="s1ab-%s-" % args.model))
    struct_dir, msa_dir = work / "out", work / "msa"
    struct_dir.mkdir(parents=True)
    msa_dir.mkdir(parents=True)

    # Same fixture the RF3 ladder and the cross-model latch census fold: CDK2 (1HCL) tiled to N aa,
    # one chain, no MSA. A refusal here is on the shapes those screens already cover.
    inp = work / ("cdk2_%d.yaml" % args.aa)
    inp.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n"
                   % cdk2(args.aa))

    cfg = _build_cfg(args.model, SPECS.get(args.model, {}), struct_dir, msa_dir)
    cfg["recycling_steps"] = args.recycling_steps
    cfg["sampling_steps"] = args.sampling_steps
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("s1ab", cfg)
    state.pfn = lambda *a, **k: None
    if cfg["model"] == "boltz2":
        state.model.progress_fn = lambda *a, **k: None

    def set_arm(arm: str) -> None:
        tt._FP32_SOFTMAX_L1_ANY_CORES = (arm == "B")
        tt._fp32_softmax_l1_plan.cache_clear()
        tt._fp32_softmax_core_grid.cache_clear()
        # a refusal recorded under one arm must not narrow the other's shape class
        tt._FP32_SOFTMAX_L1_ROW_CAP.clear()
        for key in tt.FP32_SOFTMAX_STATS:
            tt.FP32_SOFTMAX_STATS[key] = 0

    folds = []
    order = "AB" + args.arms          # one cold pair, discarded
    for i, arm in enumerate(order):
        for f in struct_dir.rglob("*"):
            if f.is_file():
                f.unlink()
        set_arm(arm)
        t0 = time.perf_counter()
        metrics, _best, _feats = state.predict_one(inp, dict(cfg, struct_dir=str(struct_dir)))
        wall = time.perf_counter() - t0
        rec = {"i": i, "arm": arm, "cold": i < 2, "fold_s": round(wall, 3),
               "digest": digest_dir(struct_dir),
               "fp32_softmax": dict(tt.FP32_SOFTMAX_STATS),
               "metrics": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                           for k, v in dict(metrics or {}).items()}}
        folds.append(rec)
        print("%2d %s%s %8.3f s  digest %s  l1 %d/%d cores %d blocks %d/%d refused %d"
              % (i, arm, " (cold)" if rec["cold"] else "       ", wall, rec["digest"],
                 rec["fp32_softmax"]["l1"], rec["fp32_softmax"]["calls"],
                 rec["fp32_softmax"]["l1_cores"], rec["fp32_softmax"]["l1_blocks"],
                 rec["fp32_softmax"]["blocks"], rec["fp32_softmax"]["l1_refused"]), flush=True)

    warm = [f for f in folds if not f["cold"]]
    med = {}
    for arm in sorted({f["arm"] for f in warm}):
        ts = [f["fold_s"] for f in warm if f["arm"] == arm]
        med[arm] = {"n": len(ts), "folds_s": ts, "median_s": statistics.median(ts),
                    "spread_pct": round(100 * (max(ts) - min(ts)) / min(ts), 2) if ts else None}
    digests = sorted({f["digest"] for f in warm})
    rep = {"model": args.model, "aa": args.aa, "arms": args.arms,
           "recycling_steps": args.recycling_steps, "sampling_steps": args.sampling_steps,
           "grid": list(tt.COMPUTE_GRID_MAIN), "folds": folds, "per_arm": med,
           "warm_digests": digests, "bit_exact": len(digests) == 1}
    if "A" in med and "B" in med:
        rep["speedup_B_over_A"] = round(med["A"]["median_s"] / med["B"]["median_s"], 4)
    Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")

    print("--- %s %d aa, grid %s" % (args.model, args.aa, rep["grid"]))
    for arm, m in med.items():
        print("  arm %s  median %8.3f s  n=%d  spread %.2f %%" % (arm, m["median_s"], m["n"],
                                                                 m["spread_pct"]))
    if "speedup_B_over_A" in rep:
        print("  B/A speedup %.4fx" % rep["speedup_B_over_A"])
    print("  warm digests: %s  -> %s" % (digests, "BIT-EXACT" if rep["bit_exact"] else "DIFFER"))


if __name__ == "__main__":
    main()
