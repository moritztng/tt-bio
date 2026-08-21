#!/usr/bin/env python3
"""One RF3 fold of a named `perf/size512` fixture, with the CIF kept for an accuracy A/B.

Step 4 of the root-cause pass needs an accuracy verdict on the two non-bit-exact levers, and
`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity` rules out scoring it on
`cdk2x2_512`: that fixture's unconstrained inter-domain hinge saturates RMSD near 8 A for any
non-bit-exact change, so it cannot separate a real accuracy loss from a bit-level one. The 298 aa
fixture is a single domain and does separate them.

Same fold path as `perf/rf3/page512_tt.py` (`tt_baseline.build_fold` -> `predict_one`) at RF3's
own shipped 10 recycles / 50 sampling steps, so the coordinates are what a user gets. Two folds:
the cold one is discarded, the warm one is the sample. Both arms run at the same seed, so a
bit-exact change would land at exactly 0.

    fold_fix_ab.py --fix cdk2x2_298 --label def --outdir perf/rf3/accuracy/298_def
"""
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", default="cdk2x2_298")
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=1, help="warm folds after the discarded cold one")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps

    assert Path(T.__file__).resolve().is_relative_to(ROOT), \
        f"tt_bio resolves to {T.__file__}, not this checkout -- set PYTHONPATH"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "rf3")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "rf3")
    assert (B.RECYCLING_STEPS, B.SAMPLING_STEPS) == (10, 50), \
        f"rf3 defaults are {B.RECYCLING_STEPS}/{B.SAMPLING_STEPS}, expected 10/50"

    tgt = a.fixdir / f"{a.fix}.yaml"
    a3m = a.fixdir / f"{a.fix}.a3m"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    one_fold, meta, state = B.build_fold(
        "rf3", ROOT / f".msa_rf3_{a.fix}", tgt, a3m)
    struct_dir = Path(meta["struct_dir"])
    a.outdir.mkdir(parents=True, exist_ok=True)

    res = {"label": a.label, "fix": a.fix, "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "diffusion_samples": B.DIFFUSION_SAMPLES, "seed": B.SEED,
           "n_msa": meta.get("n_msa"), "sha256_target": sha(tgt), "sha256_a3m": sha(a3m),
           "env_flags": {k: v for k, v in sorted(os.environ.items())
                         if k.startswith("TT_BIO_")},
           "folds": []}

    def one(tag, keep):
        fold_s, m = one_fold()
        assert m.get("msa"), f"{tag}: fold ran without an MSA -- cache seeding failed"
        cifs = sorted(struct_dir.glob("*.cif"))
        rec = {"tag": tag, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "ptm": m.get("ptm"), "n_tokens": m.get("n_tokens"), "n_atoms": m.get("n_atoms"),
               "msa": m.get("msa"),
               "cif_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                              for p in cifs},
               "opm_small_depth_stats": list(T.OPM_SMALL_DEPTH_STATS),
               "triatt_fused_hifi_stats": dict(T.TRIATT_FUSED_HIFI_STATS)}
        if keep:
            for p in cifs:
                shutil.copy2(p, a.outdir / p.name)
            rec["kept"] = [p.name for p in cifs]
        print(f"[{a.label}] {tag} {rec['fold_s']:.3f}s plddt={rec['plddt']} "
              f"ptm={rec['ptm']} cif={list(rec['cif_sha256'].values())}", flush=True)
        return rec

    res["cold"] = one("cold", keep=False)
    for i in range(a.repeat):
        res["folds"].append(one(f"warm{i}", keep=True))
    (a.outdir / "fold.json").write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", a.outdir / "fold.json", flush=True)
    state.reset()
    T.cleanup()


if __name__ == "__main__":
    main()
