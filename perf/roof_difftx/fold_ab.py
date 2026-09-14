#!/usr/bin/env python3
"""Boltz-2 512 aa fold-level A/B of the token DiT conditioning hoist, paired and interleaved ABBA.

Arm `off` is main. Arm `on` sets `tenstorrent._B2_DIT_COND_HOIST`, which is read at CALL time, so
both arms run out of one process against one loaded model and the only thing that changes between
them is whether `DiffusionTransformer.__call__` computes the 24 layers' conditioning projections
itself or takes them from two concatenated matmuls it ran once for the step.

The transform is NOT bit-exact -- the layer_norm gain moves onto the weight -- so the bar is the
structure, not a CIF sha256: CA-lDDT and Kabsch RMSD against the paired `off` arm, with the
arm's own determinism control (off against off) beside them as the floor those numbers sit on.

Protocol, fixtures and cfg come from perf/b2x-flag-levers/ab_flag_levers.py, the fold the rest of
this campaign times. 200 sampling steps, 3 recycles, full MSA: no work is removed.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, shutil, socket, statistics as st
import sys, tempfile, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "other512"))
_spec = importlib.util.spec_from_file_location(
    "_b2x_flaglev", REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py")
LEV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEV)
from cif_rmsd import read_atoms                                               # noqa: E402

FIX = REPO / "perf" / "size512" / "fixtures"
ORDER = ["off", "on", "on", "off"]           # ABBA, so box drift cancels inside a rep
OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


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


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--fixture", default="cdk2x2_512",
                    help="comma-separated; the ladder runs in ONE process on ONE loaded model, "
                         "so a size sweep pays the model load once and the arms stay paired "
                         "inside each size")
    ap.add_argument("--reps-per", default="",
                    help="comma-separated reps, one per fixture; falls back to --reps")
    args = ap.parse_args()
    OUT_PATH = args.out
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = args.steps, args.recycles

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")
    assert "TT_BIO_DIT_COND_HOIST" not in os.environ, "pinned; the arms are set in-process"
    assert TT._B2_DIT_COND_HOIST is False, "this tree must default the hoist OFF"

    dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "fixture": args.fixture,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "core_grid_main": [TT.CORE_GRID_MAIN.x, TT.CORE_GRID_MAIN.y],
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip()}
    dump()

    fixtures = [x for x in args.fixture.split(",") if x]
    reps_per = [int(x) for x in args.reps_per.split(",") if x] or [args.reps] * len(fixtures)
    assert len(reps_per) == len(fixtures), "reps-per must match fixture count"

    work = Path(tempfile.mkdtemp(prefix="roof-difftx-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    keep_dir = work / "keep"; keep_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for fx in fixtures:
        LEV._seed_msa(FIX / f"{fx}.yaml", (FIX / f"{fx}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("roof-difftx-cond-hoist", cfg)
    diff_mod = state.model.structure_module.score_model

    def fold(arm: str, tag: str, fixture: str) -> dict:
        TT._B2_DIT_COND_HOIST = arm == "on"
        try:
            diff_mod.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(FIX / f"{fixture}.yaml", cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        kept = keep_dir / f"{tag}.cif"
        shutil.copyfile(cifs[0], kept)
        return {"arm": arm, "tag": tag, "fixture": fixture,
                "fold_s": round(wall, 3), "cif": str(kept),
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2)}

    OUT["by_fixture"] = {}
    for fx, nrep in zip(fixtures, reps_per):
        runs = []
        for arm in ("off", "on"):
            r = fold(arm, f"{fx}_warm_{arm}", fx); r["warmup"] = True; runs.append(r)
            print(f"  {fx} warm {arm:3s} {r['fold_s']:7.3f}s plddt={r['plddt']} "
                  f"cif {r['cif_sha256'][:16]}", flush=True)
            OUT["by_fixture"].setdefault(fx, {})["runs"] = runs; dump()
        for i in range(nrep):
            for j, arm in enumerate(ORDER):
                r = fold(arm, f"{fx}_r{i}_{j}_{arm}", fx)
                r["warmup"] = False; r["rep"] = i; runs.append(r)
                print(f"  {fx} rep{i} {arm:3s} {r['fold_s']:7.3f}s plddt={r['plddt']} "
                      f"load={r['loadavg1']} cif {r['cif_sha256'][:16]}", flush=True)
                OUT["by_fixture"][fx]["runs"] = runs; dump()

        timed = [r for r in runs if not r["warmup"]]
        med = {a: st.median([r["fold_s"] for r in timed if r["arm"] == a]) for a in ("off", "on")}
        rep: dict = {}
        for r in timed:
            rep.setdefault(r["rep"], []).append(r)
        ab, aa = [], []
        for i in sorted(rep):
            g = rep[i]
            assert [x["arm"] for x in g] == ORDER, [x["arm"] for x in g]
            off, on = [g[0]["fold_s"], g[3]["fold_s"]], [g[1]["fold_s"], g[2]["fold_s"]]
            ab += [round(o / n, 5) for o, n in zip(off, on)]
            aa += [round(max(off) / min(off), 5), round(max(on) / min(on), 5)]
        shas = {a: sorted({r["cif_sha256"] for r in timed if r["arm"] == a})
                for a in ("off", "on")}
        offs = [r for r in timed if r["arm"] == "off"]
        ons = [r for r in timed if r["arm"] == "on"]
        e = OUT["by_fixture"][fx]
        e["median_fold_s"] = med
        e["ratio_median"] = round(med["off"] / med["on"], 5)
        e["ab_paired_ratios"] = ab
        e["ab_paired_median"] = round(st.median(ab), 5)
        e["ab_paired_all_positive"] = all(x > 1 for x in ab)
        e["aa_floor_max"] = max(aa) if aa else None
        e["aa_ratios"] = aa
        e["resolved_above_aa_floor"] = bool(aa) and st.median(ab) > max(aa)
        e["plddt"] = {a: sorted({r["plddt"] for r in timed if r["arm"] == a})
                      for a in ("off", "on")}
        e["cif_sha256"] = shas
        e["bit_exact"] = len(shas["off"]) == 1 and shas["off"] == shas["on"]
        e["structure"] = {
            "control_off_vs_off": compare(offs[0]["cif"], offs[-1]["cif"]) if len(offs) > 1
            else None,
            "on_vs_off": compare(offs[0]["cif"], ons[0]["cif"])}
        dump()
        print(f"  == {fx}: {e['ab_paired_median']}x  A/A {e['aa_floor_max']}  "
              f"resolved={e['resolved_above_aa_floor']}  "
              f"CA-RMSD {e['structure']['on_vs_off']['kabsch_rmsd_A']} A  "
              f"lDDT {e['structure']['on_vs_off']['ca_lddt']}", flush=True)

    print(json.dumps(OUT["by_fixture"], indent=1, default=str)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
