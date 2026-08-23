#!/usr/bin/env python3
"""Interleaved fold A/B of the trimul K-block width at a prime tile count, on any pair-track model.

`in0_block_w` has to divide Kt, and the shipped picker takes the widest divisor at or below 10. At a
prime Kt above 10 that band holds nothing but 1, so 544 aa (Kt = 17) runs the trimul matmul with no
K blocking at all: 19.97 TFLOP/s against 512 aa's 37.01. Kt itself always divides Kt, and its
circular buffers fit, so arm B is `in0_block_w = Kt` at exactly those lengths -- 352, 416, 544, 608,
736, 928 and 992 aa. Every other length keeps the tuned divisor byte for byte and is not an A/B at
all, which is why `--aa` has to be one of those.

The K-block width is the one matmul parameter that can change the ORDER partial sums accumulate in,
so a digest match here is a result and not a restatement of the design. Read the digest line first.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 python3 perf/trimul_kernel/kt_prime_ab.py \
        --model openfold3 --aa 544 --arms ABABAB --out /tmp/kt17_of3_544.json
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
                    help="fold order, A = the shipped divisor band, B = in0_block_w = Kt.")
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

    work = Path(tempfile.mkdtemp(prefix="kt17ab-%s-" % args.model))
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
    state.bind_run("kt17ab", cfg)
    state.pfn = lambda *a, **k: None
    if cfg["model"] == "boltz2":
        state.model.progress_fn = lambda *a, **k: None

    shipped = tt._trimul_in0_block_w

    def widened_in0_block_w(seq_len_tiles):
        """Arm B: the widest divisor of Kt whose circular buffers fit, so Kt itself at a prime Kt.

        `matmul_multi_core_reuse_mcast` double-buffers in0 and in1 one K block at a time and holds
        the output block and its fp32 accumulator whole. At Kt=17 on an 11x10 grid that is
        2*(2+2)*17 tiles + 4*(2048+4096) = 303104 B against 1461760 per bank, so the wide block is
        legal by a wide margin -- the shipped picker declines it on the divisor band, not on space.
        """
        gx, gy = tt.COMPUTE_GRID_MAIN
        pcm, pcn = -(-seq_len_tiles // gy), -(-seq_len_tiles // gx)
        budget = tt._l1_bank_bytes()
        for w in range(seq_len_tiles, 0, -1):
            if seq_len_tiles % w:
                continue
            if 2 * (pcm + pcn) * w * 2048 + pcm * pcn * (2048 + 4096) <= budget:
                return w
        return 1

    def set_arm(arm: str) -> None:
        tt._trimul_in0_block_w = widened_in0_block_w if arm == "B" else shipped
        tt._triangle_mul_program_config.cache_clear()
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
               "in0_block_w": {kt: tt._triangle_mul_program_config(kt).in0_block_w
                               for kt in ((args.aa + 31) // 32,)},
               "digest": digest_dir(struct_dir),
               "fp32_softmax": dict(tt.FP32_SOFTMAX_STATS),
               "metrics": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                           for k, v in dict(metrics or {}).items()}}
        folds.append(rec)
        print("%2d %s%s %8.3f s  digest %s  in0_block_w %s"
              % (i, arm, " (cold)" if rec["cold"] else "       ", wall, rec["digest"],
                 rec["in0_block_w"]), flush=True)

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
