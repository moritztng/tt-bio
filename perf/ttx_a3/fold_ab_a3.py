#!/usr/bin/env python3
"""Paired ABBA fold A/B for the above-cap fused triangle attention, one process, one device open.

`fold_parity_a3.py` runs one arm per process, which is right for the CIF comparison and wrong for
the ratio: three sequential 1536 aa folds on qb2 read 200.1 / 206.5 / 255.2 s with two of the three
being the SAME arm, so the session's own drift is 27.6 % and no ratio survives it. This loads the
model once and alternates the arm inside the process, ABBA per rep, so every ratio is paired and
the two same-arm folds in each rep give the A/A floor as a by-product.

`tenstorrent._SDPA_FUSED_LARGE_S` is read at call time, so no reload is needed between arms. One
warmup rep is discarded whole: each arm compiles its own kernels the first time it runs.

    TT_VISIBLE_DEVICES=2 python3 perf/ttx_a3/fold_ab_a3.py --fixture cdk2x2_1536 --reps 3 \
        --out perf/ttx_a3/abba_1536.json
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, socket, statistics as st, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "other512"))
_spec = importlib.util.spec_from_file_location(
    "_fold_parity_concat", REPO / "perf" / "roof_concat" / "fold_parity.py")
FP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FP)
LEV, FIX = FP.LEV, FP.FIX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="cdk2x2_1536")
    ap.add_argument("--reps", type=int, default=3)
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
    work = Path(tempfile.mkdtemp(prefix="ttx-a3-abba-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{a.fixture}.yaml", (FIX / f"{a.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("ttx-a3-abba", cfg)

    res = {"fixture": a.fixture, "reps": a.reps, "steps": a.steps, "recycles": a.recycles,
           "cfg_steps": cfg["sampling_steps"], "cfg_recycles": cfg["recycling_steps"],
           "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(TT.COMPUTE_GRID_MAIN),
           "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(), "folds": []}

    def fold(arm, rep, warmup):
        TT._SDPA_FUSED_LARGE_S = arm == "on"
        TS.STATS[:] = [0, 0]
        TT.SDPA_CHUNK_PICKS.clear()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(FIX / f"{a.fixture}.yaml", cfg)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        cif = sorted(struct_dir.glob("*.cif"))[0]
        rec = {"arm": arm, "rep": rep, "warmup": warmup, "fold_s": round(dt, 3),
               "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
               "cif_sha256": hashlib.sha256(cif.read_bytes()).hexdigest(),
               "served": TS.STATS[0], "declined": TS.STATS[1],
               "picks": {str(k): v for k, v in TT.SDPA_CHUNK_PICKS.items()},
               "loadavg": round(os.getloadavg()[0], 2)}
        res["folds"].append(rec)
        a.out.write_text(json.dumps(res, indent=1))
        print(json.dumps(rec), flush=True)
        return dt

    for arm in ("off", "on"):
        fold(arm, -1, True)
    for rep in range(a.reps):
        for arm in ("off", "on", "on", "off"):
            fold(arm, rep, False)

    timed = [f for f in res["folds"] if not f["warmup"]]
    def med(arm):
        return st.median([f["fold_s"] for f in timed if f["arm"] == arm])
    offs = [f["fold_s"] for f in timed if f["arm"] == "off"]
    ons = [f["fold_s"] for f in timed if f["arm"] == "on"]
    # paired per rep: the rep's two off folds against its two on folds
    ratios, aa = [], []
    for rep in range(a.reps):
        o = [f["fold_s"] for f in timed if f["arm"] == "off" and f["rep"] == rep]
        n = [f["fold_s"] for f in timed if f["arm"] == "on" and f["rep"] == rep]
        if len(o) == 2 and len(n) == 2:
            ratios.append(st.median(o) / st.median(n))
            aa.append(max(o) / min(o))
    res["summary"] = {"off_median_s": round(med("off"), 3), "on_median_s": round(med("on"), 3),
                      "median_ratio": round(st.median(ratios), 4) if ratios else None,
                      "per_rep_ratio": [round(r, 4) for r in ratios],
                      "aa_floor_per_rep": [round(x, 4) for x in aa],
                      "off_s": offs, "on_s": ons,
                      "digests": sorted({f["cif_sha256"][:16] + ":" + f["arm"] for f in timed})}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1), flush=True)
    ttnn.close_device(dev)
    return 0


sys.exit(main())
