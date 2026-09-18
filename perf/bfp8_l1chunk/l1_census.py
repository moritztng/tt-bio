#!/usr/bin/env python3
"""One 512 aa fold, instrumented for L1: which chunk every SDPA shape actually got, and every
L1 refusal the device issued.

The row's premise is that bf16-era L1 refusals bar chunks bfp8 would fit. That premise needs the
refusal sets to be NON-EMPTY at this cell before any dtype key can be worth anything, so this
measures it rather than assuming it. No timing is claimed here: a census does not need a quiet box
and this one runs co-tenanted on purpose.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bfp8-l1-chunking \
        python3 perf/bfp8_l1chunk/l1_census.py --out perf/bfp8_l1chunk/census_bf16.json
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, socket, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "_fold_parity_concat", REPO / "perf" / "roof_concat" / "fold_parity.py")
FP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FP)
LEV, FIX = FP.LEV, FP.FIX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="cdk2x2_512")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = a.steps, a.recycles

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    from tt_bio import triatt_sdpa as TS
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a_, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), _TB.__file__

    dev = get_device()
    work = Path(tempfile.mkdtemp(prefix="bfp8-l1-census-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{a.fixture}.yaml", (FIX / f"{a.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("bfp8-l1-census", cfg)

    TT.SDPA_CHUNK_PICKS.clear()
    for d in (TT.SDPA_ROUTE_COUNTS,):
        for k in d: d[k] = 0
    TS.STATS[:] = [0, 0]; TS.GATE_STATS[:] = [0, 0]
    TS.REJECTS.clear(); TS.GATE_REJECTS.clear()
    TS._PM_OVER_L1.clear(); TS._GATE_OVER_L1.clear(); TS.PM_L1_ERRORS.clear()
    TT._SDPA_Q_CHUNK_OVER_L1.clear(); TT._SDPA_QK_OVER_L1.clear(); TT._TRIATT_HIFI_OVER_L1.clear()

    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    metrics, _b, _f = state.predict_one(FIX / f"{a.fixture}.yaml", cfg)
    ttnn.synchronize_device(dev)
    dt = time.perf_counter() - t0

    cif = sorted(struct_dir.glob("*.cif"))[0]
    digest = hashlib.sha256(cif.read_bytes()).hexdigest()
    rec = {
        "cif_sha256": digest, "plddt": metrics.get("plddt") if isinstance(metrics, dict) else None,
        "fixture": a.fixture, "steps": a.steps, "recycles": a.recycles,
        "fold_s": round(dt, 3), "host": socket.gethostname(),
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "grid": list(TT.COMPUTE_GRID_MAIN),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "note": "census only -- co-tenanted, fold_s is NOT a perf claim",
        "picks": {str(k): v for k, v in TT.SDPA_CHUNK_PICKS.items()},
        "route_counts": dict(TT.SDPA_ROUTE_COUNTS),
        "fused_stats_served_declined": list(TS.STATS),
        "gate_stats_served_declined": list(TS.GATE_STATS),
        "rejects": {str(k): v for k, v in TS.REJECTS.items()},
        "gate_rejects": {str(k): v for k, v in TS.GATE_REJECTS.items()},
        "PM_OVER_L1": [str(k) for k in TS._PM_OVER_L1],
        "GATE_OVER_L1": [str(k) for k in TS._GATE_OVER_L1],
        "SDPA_Q_CHUNK_OVER_L1": [str(k) for k in TT._SDPA_Q_CHUNK_OVER_L1],
        "SDPA_QK_OVER_L1": [str(k) for k in TT._SDPA_QK_OVER_L1],
        "TRIATT_HIFI_OVER_L1": [str(k) for k in TT._TRIATT_HIFI_OVER_L1],
        "PM_L1_ERRORS": {str(k): v for k, v in TS.PM_L1_ERRORS.items()},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("rejects", "picks", "PM_L1_ERRORS")}, indent=1))
    print("PICKS:", json.dumps(rec["picks"], indent=1))
    print("REJECTS:", json.dumps(rec["rejects"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
