#!/usr/bin/env python3
"""Fold-level speed and parity of K6 -- the fused triangle-attention SDPA above the 1024 aa cap.

`tenstorrent._SDPA_FUSED_LARGE_S` makes the arm: above `triatt_sdpa._Q_SPLIT_MAX_S` it offers
`fused_pairs` ahead of the stock ladder. It is read at call time, so nothing about the loaded
model changes between the arms. "off" is what shipped before this row.

One arm per process anyway, because a fold that has already compiled the other arm's kernels
carries its program cache into the timing. `--compare` reads the directory and prints CA-lDDT,
mean |dd| and Kabsch RMSD for the three pairs that make the lever readable: the determinism
control, the lever, and the seed floor the lever has to be measured against.

K6 is NOT bit-exact by construction -- k_chunk sets the online-softmax reduction order -- so the
digest is recorded and the Angstrom figures are the verdict.

    python3 perf/ttx_a3/fold_parity_a3.py --dir perf/ttx_a3/f1536 --arm off --tag cdk2x2_1536_off
    python3 perf/ttx_a3/fold_parity_a3.py --dir perf/ttx_a3/f1536 --compare
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, shutil, socket, sys, tempfile, time
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
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--arm", choices=("off", "on"))
    ap.add_argument("--tag")
    ap.add_argument("--fixture", default="cdk2x2_1536")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    a = ap.parse_args()
    a.dir.mkdir(parents=True, exist_ok=True)
    if a.compare:
        return FP.do_compare(a.dir, a.dir / "compare.json")

    assert a.arm and a.tag, "--arm and --tag are required unless --compare"
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = a.steps, a.recycles
    if a.seed is not None:
        LEV.SEED = a.seed

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
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")

    TT._SDPA_FUSED_LARGE_S = a.arm == "on"
    TS.fused_pairs.cache_clear()
    TS.STATS[:] = [0, 0]

    dev = get_device()
    work = Path(tempfile.mkdtemp(prefix="ttx-a3-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{a.fixture}.yaml", (FIX / f"{a.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("ttx-a3-triatt-extend", cfg)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    metrics, _b, _f = state.predict_one(FIX / f"{a.fixture}.yaml", cfg)
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t0
    cifs = sorted(struct_dir.glob("*.cif"))
    assert cifs, "no CIF written"
    shutil.copyfile(cifs[0], a.dir / f"{a.tag}.cif")
    rec = {"arm": a.arm, "tag": a.tag, "fixture": a.fixture, "seed": LEV.SEED,
           "steps": a.steps, "recycles": a.recycles, "fold_s": round(wall, 3),
           "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
           "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
           "fused_large_s": TT._SDPA_FUSED_LARGE_S, "q_split_max_s": TS._Q_SPLIT_MAX_S,
           "triatt_served": TS.STATS[0], "triatt_declined": TS.STATS[1],
           "pm_over_l1": sorted(str(k) for k in TS._PM_OVER_L1),
           "pm_l1_errors": {str(k): str(v)[:200] for k, v in TS.PM_L1_ERRORS.items()},
           "sdpa_picks": {str(k): v for k, v in TT.SDPA_CHUNK_PICKS.items()},
           "grid": list(TT.COMPUTE_GRID_MAIN), "grid_measured": TT.COMPUTE_GRID_MEASURED,
           "maxrss_mb": round(int(next(l for l in open("/proc/self/status")
                                       if l.startswith("VmHWM")).split()[1]) / 1024, 1),
           "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "loadavg": os.getloadavg(),
           "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip()}
    (a.dir / f"{a.tag}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
