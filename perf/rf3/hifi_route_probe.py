#!/usr/bin/env python3
"""Does RF3's SHIPPED fused triangle-attention route actually take a non-default compute config?

Not an accuracy or a perf reading. One short fold per arm (1 recycle, 2 denoise steps) at a fixed
size, with every route counter dumped, so the exec pass knows before it spends a card-hour whether
the arm fires at all and whether `fp32_dest_acc` gets refused by L1 and silently falls to the stock
op. A lever on a route that serves 3 % of calls is dark, not neutral.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckcs", default="none,HiFi4|0|1,HiFi2|1|1")
    ap.add_argument("--recycling_steps", type=int, default=1)
    ap.add_argument("--sampling_steps", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)

    from perf_regression import SPECS, _build_cfg
    sys.path.insert(0, str(ROOT / "perf" / "rf3"))
    from make_inputs import cdk2

    from tt_bio import tenstorrent as tt
    from tt_bio import triatt_sdpa as pm
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

    work = Path(tempfile.mkdtemp(prefix="rf3-hifi-probe-%d-" % args.aa))
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
    state.bind_run("hifiprobe", cfg)
    state.pfn = lambda *a, **k: None

    rows = []
    for spec in args.ckcs.split(","):
        pm.STATS[0] = pm.STATS[1] = 0
        pm.REJECTS.clear()
        tt.SDPA_ROUTE_COUNTS["fused"] = tt.SDPA_ROUTE_COUNTS["stock"] = 0
        tt.SDPA_CHUNK_PICKS.clear()
        tt.SDPA_HIFI_CALLS[0] = 0
        tt.SDPA_RAGGED_SITES.clear()
        pm._CKC_OVERRIDE = None if spec == "none" else pm.ckc_from_env(spec.replace("|", ","))
        for f in struct_dir.rglob("*"):
            if f.is_file():
                f.unlink()
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(inp, dict(cfg, struct_dir=str(struct_dir)))
        wall = time.perf_counter() - t0
        rec = {
            "ckc": spec,
            "resolved_ckc": (None if pm._CKC_OVERRIDE is None
                             else [str(pm._CKC_OVERRIDE[0]), bool(pm._CKC_OVERRIDE[1]),
                                   bool(pm._CKC_OVERRIDE[2])]),
            "fold_s": round(wall, 3),
            "n_tokens": (metrics or {}).get("n_tokens"),
            "pm_served": pm.STATS[0], "pm_declined": pm.STATS[1],
            "pm_rejects": {"%s %s" % (k[0], list(k[1])): v for k, v in pm.REJECTS.items()},
            "pm_over_l1": ["%s" % (list(k),) for k in pm._PM_OVER_L1],
            "pm_l1_errors": {str(list(k)): v[:200] for k, v in pm.PM_L1_ERRORS.items()},
            "route_counts": dict(tt.SDPA_ROUTE_COUNTS),
            "chunk_picks": {"%dx%d" % k: v for k, v in tt.SDPA_CHUNK_PICKS.items()},
            "sdpa_hifi_calls": tt.SDPA_HIFI_CALLS[0],
            "ragged_sites": {k: list(v) for k, v in tt.SDPA_RAGGED_SITES.items()},
            "fp32_softmax_stats": dict(tt.FP32_SOFTMAX_STATS),
        }
        rows.append(rec)
        print(json.dumps(rec), flush=True)
        Path(args.out).write_text(json.dumps({"aa": args.aa, "rows": rows}, indent=1) + "\n")

    Path(args.out).write_text(json.dumps(
        {"aa": args.aa, "host": os.uname().nodename,
         "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": list(tt.COMPUTE_GRID_MAIN),
         "recycling_steps": args.recycling_steps, "sampling_steps": args.sampling_steps,
         "pairformer_flags": {k: repr(v) for k, v in PAIRFORMER_FLAGS.items()},
         "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
