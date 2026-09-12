#!/usr/bin/env python3
"""`cdk2x2_298` per bfloat8_b SITE, scored against the campaign's kill/hold/pass bar.

One process, one device open, one arm per site plus a `base` arm in bf16 and (with more than one
site) the union of every site. Each arm folds the monomeric control and, with `--seq512`, the
512 aa cell as well, since precision effects are size-dependent and 298 aa is the monomeric
control, not the cell.

Two references, deliberately:

* the in-process `base` arm, which is what every site's delta is measured against. Same host,
  same card, same build, so the difference is the site and nothing else.
* `perf/b2x-baseline-attrib/ref/main20260911_*_cdk2x2_298.cif`, the campaign's committed
  reference, taken on Blackhole. `base` scored against it is the WH-vs-BH offset of an unchanged
  fold, and every site's number has to be read on top of that, not instead of it.

Bar (the campaign's, not this task's): >0.60 A all-atom kills the site, 0.35-0.60 A holds it for
a decision, <=0.35 A passes. `cdk2x2_512` cannot score a non-bit-exact change (memory
`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`), so at 512 aa the CIF sha256 and
plDDT are recorded as a did-it-change check and the RMSD is reported against `base` only as a
magnitude, never as a verdict.
"""
from __future__ import annotations

import argparse, hashlib, json, os, socket, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prodcfg                                                              # noqa: E402
from prodcfg import FIX, REPO, RECYCLES, SAMPLES, SEED, STEPS               # noqa: E402
from prodcfg import assert_checkout                                         # noqa: E402

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "other512"))

REF_DIR = REPO / "perf" / "b2x-baseline-attrib" / "ref"

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def verdict(rmsd: float) -> str:
    return "pass" if rmsd <= 0.35 else ("hold" if rmsd <= 0.60 else "reject")


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--sites", default="")
    ap.add_argument("--union", action=argparse.BooleanOptionalAction, default=True,
                    help="also fold the union of --sites; off when --sites is one shard of a "
                         "fan-out, where the union of a shard means nothing")
    ap.add_argument("--seq512", action="store_true",
                    help="also fold the 512 aa cell per arm (roughly triples the wall clock)")
    ap.add_argument("--cifdir", type=Path, default=None)
    args = ap.parse_args()
    OUT_PATH = args.out

    import numpy as np
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio import tenstorrent as T
    from tt_bio.worker import _WorkerState
    from tt_bio import esmfold2 as _E
    from cif_rmsd import read_atoms, kabsch_rmsd
    _E.set_progress(lambda *a, **k: None)

    sites = [s for s in args.sites.split(",") if s] or list(T.PAIR_B8_SITES)
    unknown = [s for s in sites if s not in T.PAIR_B8_SITES]
    assert not unknown, f"unknown site(s) {unknown}; known {list(T.PAIR_B8_SITES)}"

    arms: list[tuple[str, frozenset]] = [("base", frozenset())]
    arms += [(s, frozenset([s])) for s in sites]
    if len(sites) > 1 and args.union:
        arms.append(("all", frozenset(sites)))

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": [g.x, g.y], "arch": str(dev.arch()),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tt_bio": assert_checkout(),
        "protocol": {"recycling_steps": RECYCLES, "sampling_steps": args.steps,
                     "diffusion_samples": SAMPLES, "seed": SEED},
        "bar": "<=0.35 pass / 0.35-0.60 hold / >0.60 reject, all-atom on cdk2x2_298",
        "arms": {a: sorted(ss) for a, ss in arms},
    }
    try:
        import importlib.metadata as _md
        OUT["env"]["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z-bfp8-parity-"))
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    targets = ["cdk2x2_298"] + (["cdk2x2_512"] if args.seq512 else [])
    for n in targets:
        prodcfg.seed_msa(FIX / f"{n}.yaml", (FIX / f"{n}.a3m").read_text(), msa_dir)

    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    cfg = prodcfg.build_cfg(msa_dir, struct_dir, steps=args.steps)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z-bfp8-narrow", cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t0, 2)
    dump()

    def set_arm(site_set: frozenset):
        """Set the whole site set explicitly, so no arm inherits the previous arm's."""
        T._PAIR_B8_SITES = site_set
        for m in state.model.modules():
            gp = getattr(m, "_gp_cache", None)
            if isinstance(gp, dict):
                gp.clear()
                m._gp_bias_cache.clear()

    def fold(target: Path, out_dir: Path):
        for p in out_dir.glob("*"):
            p.unlink()
        c = prodcfg.build_cfg(msa_dir, out_dir, steps=args.steps)
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, c)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        cif = sorted(out_dir.glob("*.cif"))
        assert cif, f"no CIF written for {target.name}"
        return {"fold_s": round(wall, 3),
                "cif_sha256": hashlib.sha256(cif[0].read_bytes()).hexdigest()[:16],
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif": cif[0].read_text()}

    cifs = args.cifdir or (work / "cif")
    cifs.mkdir(parents=True, exist_ok=True)
    folds: dict = {}
    # A site that cannot fold is a result against that site, not a crash of the sweep: a storage
    # dtype changes which program config a tuned matmul picks, and that alone can overflow L1.
    errors: dict = {}
    for name, ss in arms:
        set_arm(ss)
        folds[name] = {}
        for tgt in targets:
            try:
                f = fold(FIX / f"{tgt}.yaml", struct_dir)
            except Exception as e:                                    # noqa: BLE001
                msg = f"{type(e).__name__}: {e}".split("\nbacktrace")[0][:600]
                errors.setdefault(name, {})[tgt] = msg
                OUT["errors"] = errors
                dump()
                print(f"  {tgt} {name:12s} FAILED  {msg[:160]}", flush=True)
                continue
            (cifs / f"{name}__{tgt}.cif").write_text(f.pop("cif"))
            folds[name][tgt] = f
            OUT["folds"] = folds
            dump()
            print(f"  {tgt} {name:12s} {f['fold_s']:7.3f} s  sha {f['cif_sha256']}  "
                  f"plddt {f['plddt']}", flush=True)
    set_arm(frozenset())
    if not folds.get("base"):
        OUT["summary"] = {"error": "the base arm did not fold; nothing to score against"}
        dump()
        return 1

    def score(path_a: Path, path_b: Path):
        ka, xa = read_atoms(path_a)
        kb, xb = read_atoms(path_b)
        if ka != kb:
            return {"error": "atom identity differs; cannot compare by order",
                    "n_a": len(ka), "n_b": len(kb)}
        ca = np.array([k[2] == "CA" for k in ka]) if ka and len(ka[0]) > 2 else None
        out = {"all_atom_rmsd_A": round(kabsch_rmsd(xa, xb), 6), "n_atoms": len(xa)}
        if ca is not None and ca.any():
            out["ca_rmsd_A"] = round(kabsch_rmsd(xa[ca], xb[ca]), 6)
            out["n_ca"] = int(ca.sum())
        return out

    # The committed Blackhole reference, so `base`'s own WH-vs-BH offset is visible and every
    # site's number is read on top of it rather than instead of it.
    committed = {t: next(iter(sorted(REF_DIR.glob(f"main*_{t}.cif"))), None) for t in targets}
    OUT["committed_ref"] = {t: (p.name if p else None) for t, p in committed.items()}

    results = {}
    for name, _ss in arms:
        r = {}
        for tgt in targets:
            if tgt not in folds.get(name, {}) or tgt not in folds["base"]:
                r[tgt] = {"error": errors.get(name, {}).get(tgt, "not folded")}
                continue
            mine = cifs / f"{name}__{tgt}.cif"
            base = cifs / f"base__{tgt}.cif"
            e = {"vs_base": score(mine, base),
                 "cif_sha256": folds[name][tgt]["cif_sha256"],
                 "plddt": folds[name][tgt]["plddt"],
                 "fold_s": folds[name][tgt]["fold_s"],
                 "bit_exact_vs_base": folds[name][tgt]["cif_sha256"]
                 == folds["base"][tgt]["cif_sha256"]}
            if committed.get(tgt):
                e["vs_committed_bh_ref"] = score(mine, committed[tgt])
            if tgt == "cdk2x2_298" and "all_atom_rmsd_A" in e["vs_base"]:
                e["verdict"] = verdict(e["vs_base"]["all_atom_rmsd_A"])
            r[tgt] = e
        results[name] = r
        OUT["parity"] = results
        dump()

    OUT["summary"] = {
        name: ({"error": r["cdk2x2_298"]["error"]} if "error" in r["cdk2x2_298"] else
               {"all_atom_A_298": r["cdk2x2_298"]["vs_base"].get("all_atom_rmsd_A"),
                "verdict": r["cdk2x2_298"].get("verdict"),
                "plddt_298": r["cdk2x2_298"]["plddt"],
                "plddt_512": r.get("cdk2x2_512", {}).get("plddt")})
        for name, r in results.items()}
    dump()
    print(json.dumps(OUT["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
