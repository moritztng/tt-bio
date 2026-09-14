#!/usr/bin/env python3
"""Fold-level accuracy of the one-op head re-assembly: one arm per process, compared offline.

The lever is decided when the module is built, not per call -- the gate and output projection are
a different SHAPE under it -- so the two arms cannot share a loaded model the way a call-time flag
can. Each invocation loads the model once with `_APB_CONCAT_HEADS` pinned, folds one fixture at a
fixed seed, and keeps the CIF. `--compare` then reads a directory of them.

Three things have to be in that directory for the comparison to mean anything: the two arms, the
same arm twice (the determinism control, which is the floor these numbers sit on), and the same
arm at a different seed (the seed floor, which is the variation already accepted).
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, shutil, socket, sys, tempfile, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "other512"))
_spec = importlib.util.spec_from_file_location(
    "_b2x_flaglev", REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py")
LEV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEV)
from cif_rmsd import read_atoms                                              # noqa: E402

FIX = REPO / "perf" / "size512" / "fixtures"


def ca(p):
    keys, xyz = read_atoms(Path(p))
    idx = [i for i, k in enumerate(keys) if k[2] == "CA"]
    return np.array([xyz[i] for i in idx]), [keys[i] for i in idx]


def kabsch(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt((((a @ r.T) - b) ** 2).sum(1).mean()))


def compare(pa, pb):
    """CA-lDDT (superposition free, the reading with teeth) and a global Kabsch RMSD."""
    A, ka = ca(pa)
    B, kb = ca(pb)
    assert ka == kb, "different residue sets"
    n = len(A)
    da = np.linalg.norm(A[:, None] - A[None], axis=-1)
    db = np.linalg.norm(B[:, None] - B[None], axis=-1)
    m = (da < 15.0) & ~np.eye(n, dtype=bool)
    diff = np.abs(da - db)[m]
    return {"ca_atoms": n,
            "ca_lddt": round(float(np.mean([(diff <= t).mean()
                                            for t in (0.5, 1.0, 2.0, 4.0)]) * 100), 4),
            "mean_abs_dd_A": round(float(diff.mean()), 4),
            "kabsch_rmsd_A": round(kabsch(A, B), 4)}


def do_compare(d: Path, out: Path) -> int:
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))
            if p.name != out.name}
    have = {k: d / f"{k}.cif" for k in runs if (d / f"{k}.cif").is_file()}
    res = {"runs": runs, "pairs": {}}
    def pair(a, b, name):
        if a in have and b in have:
            res["pairs"][name] = compare(have[a], have[b])
    for fx in sorted({r["fixture"] for r in runs.values()}):
        t = lambda s: f"{fx}_{s}"                                            # noqa: E731
        pair(t("off"), t("off2"), f"{fx}: control off vs off (determinism)")
        pair(t("off"), t("on"), f"{fx}: on vs off (the lever)")
        pair(t("off"), t("offseed"), f"{fx}: off vs off at another seed (seed floor)")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["pairs"], indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--arm", choices=("off", "on"))
    ap.add_argument("--tag")
    ap.add_argument("--fixture", default="cdk2x2_512")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    a = ap.parse_args()
    a.dir.mkdir(parents=True, exist_ok=True)
    if a.compare:
        return do_compare(a.dir, a.dir / "compare.json")

    assert a.arm and a.tag, "--arm and --tag are required unless --compare"
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = a.steps, a.recycles
    if a.seed is not None:
        LEV.SEED = a.seed

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a_, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")
    TT._APB_CONCAT_HEADS = a.arm == "on"
    TT.APB_CONCAT_HEADS_STATS[:] = [0, 0]

    dev = get_device()
    work = Path(tempfile.mkdtemp(prefix="roof-concat-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{a.fixture}.yaml", (FIX / f"{a.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("roof-concat-heads", cfg)
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
           "concat_heads_stats": list(TT.APB_CONCAT_HEADS_STATS),
           "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "loadavg": os.getloadavg(),
           "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip()}
    (a.dir / f"{a.tag}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
