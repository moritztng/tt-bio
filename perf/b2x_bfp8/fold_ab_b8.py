#!/usr/bin/env python3
"""Boltz-2 512 aa fold A/B: the pair track in bf16 against bfloat8_b, plus the trunk fidelity arm.

One process, one device open, arms alternated round-robin, the cold fold discarded. Every arm
sets EVERY flag explicitly, so no arm can inherit the previous one's setting and be scored as an
A/B when it was an A/A. `b16` runs `--reps` times and its own spread is this session's A/A floor.

Phase 2 folds `cdk2x2_298` -- one real CDK2 domain, no hinge -- once per arm in the SAME process,
and scores every arm against `b16` with all-atom and CA Kabsch RMSD. `cdk2x2_512` cannot score a
non-bit-exact change (memory `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`), so the
512 aa CIF sha256 and plDDT are recorded as a did-it-change check and nothing more.
"""
from __future__ import annotations

import argparse, hashlib, json, os, socket, statistics as st, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prodcfg                                                              # noqa: E402
from prodcfg import FIX, REPO, RECYCLES, SAMPLES, SEED, STEPS               # noqa: E402
from prodcfg import assert_checkout                                         # noqa: E402

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "other512"))

#: arm -> (pair track in bfloat8_b, trunk matmul fidelity)
ARMS = {
    "b16":      (False, "HiFi4"),
    "b8":       (True,  "HiFi4"),
    "hifi3":    (False, "HiFi3"),
    "b8_hifi3": (True,  "HiFi3"),
}

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()
    OUT_PATH = args.out
    arms = [a for a in args.arms.split(",") if a]

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio import tenstorrent as T
    from tt_bio.worker import _WorkerState
    from tt_bio import esmfold2 as _E
    from cif_rmsd import read_atoms, kabsch_rmsd
    import numpy as np
    _E.set_progress(lambda *a, **k: None)

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": [g.x, g.y], "arch": str(dev.arch()),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tt_bio": assert_checkout(),
        "protocol": {"recycling_steps": RECYCLES, "sampling_steps": args.steps,
                     "diffusion_samples": SAMPLES, "seed": SEED,
                     "fixture": "perf/size512/fixtures/cdk2x2_512.yaml + its 35-row a3m"},
        "arms": {a: {"pair_b8": ARMS[a][0], "trunk_fidelity": ARMS[a][1]} for a in arms},
    }
    try:
        import importlib.metadata as _md
        OUT["env"]["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2x-bfp8-"))
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for n in ("cdk2x2_512", "cdk2x2_298"):
        prodcfg.seed_msa(FIX / f"{n}.yaml", (FIX / f"{n}.a3m").read_text(), msa_dir)

    def make_cfg(struct_dir: Path) -> dict:
        return prodcfg.build_cfg(msa_dir, struct_dir, steps=args.steps)

    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    cfg = make_cfg(struct_dir)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2x-bfp8-pair-track", cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t0, 2)
    dump()

    # The trunk pairformer's compute kernel config is a distinct object per TorchWrapper, so an
    # in-place write to its fidelity provably cannot reach the diffusion or confidence stages.
    from tt_bio.tenstorrent import PairformerModule
    trunk_pf = [m for _n, m in state.model.named_modules()
                if isinstance(m, PairformerModule) and m.n_blocks == 64]
    assert len(trunk_pf) == 1, f"expected one 64-block trunk pairformer, found {len(trunk_pf)}"
    trunk_ckc = trunk_pf[0].compute_kernel_config
    OUT["trunk_ckc_default"] = str(trunk_ckc.math_fidelity)
    dump()

    def set_arm(name: str):
        """Set EVERY flag this A/B varies, so no arm inherits the previous arm's setting."""
        pair_b8, fid = ARMS[name]
        T._PAIR_B8 = pair_b8
        trunk_ckc.math_fidelity = getattr(ttnn.MathFidelity, fid)
        for m in state.model.modules():
            tm = getattr(m, "_gp_cache", None)
            if isinstance(tm, dict):
                tm.clear()
                m._gp_bias_cache.clear()

    def fold(target: Path, out_dir: Path):
        for p in out_dir.glob("*"):
            p.unlink()
        c = make_cfg(out_dir)
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, c)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        cifs = sorted(out_dir.glob("*.cif"))
        assert cifs, f"no CIF written for {target.name}"
        return {"fold_s": round(wall, 3),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16],
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif": cifs[0].read_text()}

    # ---- phase 1: timed 512 aa A/B -------------------------------------------------
    set_arm("b16")
    cold = fold(FIX / "cdk2x2_512.yaml", struct_dir)
    OUT["cold_discarded_s"] = cold["fold_s"]
    dump()
    print(f"cold {cold['fold_s']:.3f} s (discarded)", flush=True)

    runs = []
    for r in range(args.reps):
        for name in arms:
            set_arm(name)
            f = fold(FIX / "cdk2x2_512.yaml", struct_dir)
            f.pop("cif")
            f.update(arm=name, rep=r)
            runs.append(f)
            OUT["runs_512"] = runs
            dump()
            print(f"  rep{r} {name:9s} {f['fold_s']:8.3f} s  sha {f['cif_sha256']}  "
                  f"plddt {f['plddt']}", flush=True)

    base = st.median([x["fold_s"] for x in runs if x["arm"] == "b16"])
    OUT["timing_512"] = {
        a: {"fold_s": [x["fold_s"] for x in runs if x["arm"] == a],
            "median_s": round(st.median([x["fold_s"] for x in runs if x["arm"] == a]), 3),
            "speedup_vs_b16": round(base / st.median(
                [x["fold_s"] for x in runs if x["arm"] == a]), 4),
            "cif_sha256": sorted({x["cif_sha256"] for x in runs if x["arm"] == a}),
            "plddt": sorted({x["plddt"] for x in runs if x["arm"] == a})}
        for a in arms}
    b16s = [x["fold_s"] for x in runs if x["arm"] == "b16"]
    OUT["aa_floor_s"] = round(max(b16s) - min(b16s), 3)
    OUT["aa_floor_rel"] = round((max(b16s) - min(b16s)) / base, 5)
    dump()

    if args.skip_parity:
        print(json.dumps(OUT["timing_512"], indent=1))
        return 0

    # ---- phase 2: cdk2x2_298 parity control, same process ---------------------------
    pdir = work / "parity"; pdir.mkdir(parents=True)
    ctrl = {}
    for name in arms:
        set_arm(name)
        f = fold(FIX / "cdk2x2_298.yaml", pdir)
        (pdir / f"{name}.cif").write_text(f.pop("cif"))
        ctrl[name] = f
        OUT["control_298"] = ctrl
        dump()
        print(f"  298 {name:9s} {f['fold_s']:7.3f} s  sha {f['cif_sha256']}", flush=True)

    ref_keys, ref_xyz = read_atoms(pdir / f"{arms[0]}.cif")
    ca = np.array([k[2] == "CA" for k in ref_keys]) if ref_keys and len(ref_keys[0]) > 2 else None
    parity = {}
    for name in arms:
        keys, xyz = read_atoms(pdir / f"{name}.cif")
        assert keys == ref_keys, f"atom identity differs in {name}: cannot compare by order"
        parity[name] = {
            "all_atom_rmsd_A": round(kabsch_rmsd(xyz, ref_xyz), 6),
            "ca_rmsd_A": (round(kabsch_rmsd(xyz[ca], ref_xyz[ca]), 6)
                          if ca is not None and ca.any() else None),
            "n_atoms": len(xyz), "n_ca": int(ca.sum()) if ca is not None else None,
            "cif_sha256": ctrl[name]["cif_sha256"], "plddt": ctrl[name]["plddt"],
            "bit_exact_vs_b16": ctrl[name]["cif_sha256"] == ctrl[arms[0]]["cif_sha256"],
        }
    OUT["parity_298"] = {"reference_arm": arms[0], "bar": "<=0.35 pass / 0.35-0.60 hold / >0.60 reject",
                         "arms": parity}
    dump()
    print(json.dumps({"timing": OUT["timing_512"], "parity": parity}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
