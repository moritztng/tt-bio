#!/usr/bin/env python3
"""RF3 a0-vs-a1+pad whole-fold A/B at one size, interleaved inside one process.

The 512 aa cell is measured with `page512_tt.py` under the perf page`s own protocol (two
processes per arm, the page fixture, the page`s timed region). This harness closes the 768 and
1024 aa rungs, where a per-arm process would pay a second checkpoint load and a second warm-up
for a reading whose only question is which arm is faster.

The route switch is exact rather than approximate. `TriangleAttention._attend_heads` and the
s-track site both branch on `_FP32_SOFTMAX or self.fp32_softmax`, and BOTH terms are read at
call time (tenstorrent.py:4575, 4939). So a model constructed with `fp32_softmax=False` runs
a1 while the module global is off and a0`s materialised route while it is on, and the two arms
share one checkpoint, one device and one program cache.

`arms.apply_arm` is deliberately not used: it refuses exactly this combination, because for a
one-shot process a set `_FP32_SOFTMAX` would silently turn an a1 row back into a0. Here the
combination IS the instrument, so the flags are set here and the resolved route is recorded per
fold.

Arm A is a0 (shipped). Arm B is a1 + `TT_BIO_SDPA_RAGGED_PAD`. The pad flag is read at call
time too, through `_sdpa_masked`, so it is toggled with the arm; the ragged/aligned census is
recorded per fold, because a pad that fired on nothing must not read as a pad that was tested.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python3 perf/rf3/ladder_arm_ab.py \
        --aa 768 --arms ABAB --out perf/rf3/results/ladder_768.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import os
import sys
import tempfile

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
    ap.add_argument("--aa", type=int, required=True)
    ap.add_argument("--arms", default="ABAB", help="warm fold order after one discarded cold pair")
    ap.add_argument("--recycling_steps", type=int, default=10, help="RF3 ships 10")
    ap.add_argument("--sampling_steps", type=int, default=50, help="RF3 ships 50")
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
    from tt_bio.rf3.remap import PAIRFORMER_FLAGS
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts

    _E.set_progress(lambda *a, **k: None)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    assert Path(tt.__file__).resolve().is_relative_to(ROOT), \
        f"tt_bio resolves to {tt.__file__}, not this checkout -- set PYTHONPATH"
    # Build for a1 and switch with the global. The other direction does not exist: a module
    # global cannot turn a per-instance True back off.
    PAIRFORMER_FLAGS["fp32_softmax"] = False

    work = Path(tempfile.mkdtemp(prefix="rf3-arm-ab-%d-" % args.aa))
    struct_dir, msa_dir = work / "out", work / "msa"
    struct_dir.mkdir(parents=True)
    msa_dir.mkdir(parents=True)
    inp = work / ("cdk2_%d.yaml" % args.aa)
    inp.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n"
                   % cdk2(args.aa))

    cfg = _build_cfg("rf3", SPECS.get("rf3", {}), struct_dir, msa_dir)
    cfg["recycling_steps"] = args.recycling_steps
    cfg["sampling_steps"] = args.sampling_steps
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("armab", cfg)
    state.pfn = lambda *a, **k: None

    def set_arm(arm: str) -> None:
        tt._FP32_SOFTMAX = (arm == "A")
        tt._SDPA_RAGGED_PAD = (arm == "B")
        tt.SDPA_RAGGED_PAD_STATS[0] = 0
        tt.SDPA_RAGGED_SITES.clear()
        for key in tt.FP32_SOFTMAX_STATS:
            tt.FP32_SOFTMAX_STATS[key] = 0
        for key in tt.TRIATT_FUSED_HIFI_STATS:
            tt.TRIATT_FUSED_HIFI_STATS[key] = 0

    folds = []
    order = "AB" + args.arms
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
               "route": {"_FP32_SOFTMAX": bool(tt._FP32_SOFTMAX),
                         "fp32_softmax": PAIRFORMER_FLAGS["fp32_softmax"],
                         "_SDPA_RAGGED_PAD": bool(tt._SDPA_RAGGED_PAD),
                         "_TRIATT_FUSED_HIFI": bool(tt._TRIATT_FUSED_HIFI)},
               "ragged_sites": {k: list(v) for k, v in tt.SDPA_RAGGED_SITES.items()},
               "ragged_padded": tt.SDPA_RAGGED_PAD_STATS[0],
               "fp32_softmax_stats": dict(tt.FP32_SOFTMAX_STATS),
               "n_tokens": (metrics or {}).get("n_tokens"),
               "plddt": (metrics or {}).get("plddt")}
        folds.append(rec)
        print("%2d %s%s %9.3f s  digest %s  n_tok %s  ragged %s padded %d"
              % (i, arm, " (cold)" if rec["cold"] else "       ", wall, rec["digest"],
                 rec["n_tokens"], rec["ragged_sites"], rec["ragged_padded"]), flush=True)
        Path(args.out).write_text(json.dumps({"partial": True, "folds": folds}, indent=1) + "\n")

    warm = [f for f in folds if not f["cold"]]
    med = {}
    for arm in sorted({f["arm"] for f in warm}):
        ts = [f["fold_s"] for f in warm if f["arm"] == arm]
        med[arm] = {"n": len(ts), "folds_s": ts, "median_s": statistics.median(ts),
                    "spread_pct": round(100 * (max(ts) - min(ts)) / min(ts), 2) if len(ts) > 1
                    else None}
    rep = {"model": "rf3", "aa": args.aa, "arms": args.arms, "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "arm_A": "a0 shipped (materialised fp32 softmax)",
           "arm_B": "a1 + TT_BIO_SDPA_RAGGED_PAD (fused SDPA at the op default, ragged tail masked)",
           "recycling_steps": args.recycling_steps, "sampling_steps": args.sampling_steps,
           "grid": list(tt.COMPUTE_GRID_MAIN), "folds": folds, "per_arm": med}
    if "A" in med and "B" in med:
        rep["speedup_B_over_A"] = round(med["A"]["median_s"] / med["B"]["median_s"], 4)
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")

    print("--- rf3 %d aa, grid %s" % (args.aa, rep["grid"]))
    for arm, m in med.items():
        print("  arm %s  median %9.3f s  n=%d  spread %s" % (arm, m["median_s"], m["n"],
                                                            m["spread_pct"]))
    if "speedup_B_over_A" in rep:
        print("  B/A speedup %.4fx" % rep["speedup_B_over_A"])


if __name__ == "__main__":
    main()
