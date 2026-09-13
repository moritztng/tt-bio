#!/usr/bin/env python3
"""Score `TT_BIO_FUSE_BIAS_STACKS` at 512 aa the only way this fixture can be read.

Three readings, each with the thing it is compared against, because a bare RMSD on `cdk2x2_512`
means nothing: its two pseudo-domains have no interface and the hinge between them saturates
whole-molecule RMSD for any reassociation.

  per pseudo-domain     each domain superposed alone, all-atom and CA, plus the hinge angle. The
                        only structural reading the fixture supports.
  CA-lDDT               superposition-free and local, so the hinge cannot reach it (Mariani 2013,
                        inclusion radius 15 A, thresholds 0.5/1/2/4 A). Computed arm-against-arm
                        and against the experimental structure 1HCL, which no sampler argument
                        can reach: cdk2x2_512 is CDK2 followed by its own residues 1-214, so both
                        pseudo-domains have a native answer.
  the seed floor        the SAME arm at different seeds. Boltz-2's sampler is stochastic, so a
                        lever whose move is inside that spread has not been shown to do anything.

    score.py <cifdir> --runs <fold_seeds.json> --split 298 --out <json> [--arms off,on]

The arms are named on the command line, first one the baseline every other is read against, so the
same instrument scores any lever whose folds are laid out this way (`b2z2-unfused-silu-recover`
scores `base,usilu,usilu32` with it).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "other512"))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))
sys.path.insert(0, str(REPO / "perf" / "fused_sdpa"))
from cif_rmsd import kabsch_rmsd, read_atoms                             # noqa: E402
from domain_split import angle_between, kabsch                           # noqa: E402
from basin_lddt import lddt_per_residue                                  # noqa: E402
from of3_score_ref import GT_SEGMENTS, ca_map                            # noqa: E402

GT = REPO / "perf" / "fused_sdpa" / "cifs" / "1hcl.cif"


def load(cif: Path, split: int):
    keys, xyz = read_atoms(cif)
    seq = np.array([int(k[1]) for k in keys])
    ca = np.array([i for i, k in enumerate(keys) if k[2] == "CA"])
    return {"cif": cif, "keys": keys, "xyz": xyz, "seq": seq, "ca": ca,
            "d1": np.where(seq <= split)[0], "d2": np.where(seq > split)[0],
            "ca_seq": seq[ca]}


def pair(a, b, split: int) -> dict:
    """Every reading of one CIF against another. `a` is the arm, `b` the reference."""
    assert a["keys"] == b["keys"], "atom identity differs"
    out = {"whole_all_atom_A": kabsch_rmsd(a["xyz"], b["xyz"]),
           "whole_ca_A": kabsch_rmsd(a["xyz"][a["ca"]], b["xyz"][b["ca"]])}
    rots = {}
    for tag, sel in (("domain1", "d1"), ("domain2", "d2")):
        idx = a[sel]
        if not len(idx):                       # 298 aa is one pseudo-domain, not two
            continue
        R, r = kabsch(a["xyz"][idx], b["xyz"][idx])
        rots[tag] = R
        out[f"{tag}_all_atom_A"] = r
        cidx = np.array([i for i in a["ca"] if i in set(idx.tolist())], dtype=int)
        out[f"{tag}_ca_A"] = kabsch_rmsd(a["xyz"][cidx], b["xyz"][cidx])
    n1, n2 = len(a["d1"]), len(a["d2"])
    out["hinge_deg"] = angle_between(rots["domain1"], rots["domain2"]) if n2 else 0.0
    out["hinge_free_all_atom_A"] = math.sqrt(
        (n1 * out["domain1_all_atom_A"] ** 2
         + n2 * out.get("domain2_all_atom_A", 0.0) ** 2) / (n1 + n2))
    # lDDT is superposition-free, so it is the one number the hinge cannot move. Whole chain
    # first, then each domain on its own CA set.
    _pr, out["lddt_ca"] = lddt_per_residue(a["xyz"][a["ca"]], b["xyz"][b["ca"]])
    for tag, sel in (("domain1", "d1"), ("domain2", "d2")):
        m = np.isin(a["ca_seq"], a["seq"][a[sel]])
        if m.sum() > 1:
            _pr, out[f"lddt_ca_{tag}"] = lddt_per_residue(
                a["xyz"][a["ca"]][m], b["xyz"][b["ca"]][m])
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


def native(cif: Path, size: int, gt: dict) -> dict:
    """CA RMSD and CA-lDDT against the experimental structure, per pseudo-domain."""
    pm = ca_map(cif)
    out = {}
    for name, pairs in GT_SEGMENTS[size].items():
        use = [(p, q) for p, q in pairs if p in pm and q in gt]
        bad = [(p, q) for p, q in use if pm[p][0] != gt[q][0]]
        assert not bad, f"residue identity mismatch vs 1HCL: {bad[:5]}"
        A = np.array([pm[p][1] for p, _ in use])
        B = np.array([gt[q][1] for _, q in use])
        _pr, l = lddt_per_residue(A, B)
        out[name] = {"n_ca": len(use), "ca_rmsd_A": round(kabsch_rmsd(A, B), 5),
                     "lddt_ca": round(l, 5)}
    return out


def spread(vals):
    v = [x for x in vals if x is not None]
    return {"n": len(v), "min": round(min(v), 5), "max": round(max(v), 5),
            "mean": round(float(np.mean(v)), 5)} if v else {"n": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cifdir", type=Path)
    ap.add_argument("--runs", type=Path, required=True, help="fold_seeds.py --out json")
    ap.add_argument("--split", type=int, default=298, help="last label_seq_id of pseudo-domain 1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="off,on", help="first is the baseline")
    a = ap.parse_args()
    arms = a.arms.split(",")
    base = arms[0]

    runs = json.loads(a.runs.read_text())
    gt = ca_map(GT)
    report = {"cifdir": str(a.cifdir), "split_seq_id": a.split, "gt": GT.name,
              "env": runs["env"], "sizes": {}}

    by_size: dict[str, dict] = {}
    for r in runs["runs"]:
        by_size.setdefault(r["target"].split("_")[-1], {})[r["tag"]] = r

    for size, tags in by_size.items():
        n = int(size)
        split = a.split if n > a.split else n            # 298 aa is one domain, not two
        S = {t: load(next((a.cifdir / f"{size}_{t}").glob("*.cif")), split) for t in tags}
        seeds = sorted({r["seed"] for r in tags.values()})
        sec = {"n_atoms": len(next(iter(S.values()))["keys"]), "arms": sorted(tags),
               "seeds": seeds, "plddt": {t: tags[t]["plddt"] for t in sorted(tags)},
               "sha256": {t: tags[t]["sha256"] for t in sorted(tags)},
               "fold_s": {t: tags[t]["fold_s"] for t in sorted(tags)}}

        rep = [t for t in tags if t.endswith("_r1")]
        if rep:
            t = rep[0]
            sec["aa_floor"] = pair(S[t], S[t[:-3]], split)
            sec["aa_floor_bitexact"] = tags[t]["sha256"] == tags[t[:-3]]["sha256"]

        sec["lever"] = {f"{arm} seed{s}": pair(S[f"{arm}-s{s}"], S[f"{base}-s{s}"], split)
                        for arm in arms[1:] for s in seeds
                        if f"{arm}-s{s}" in S and f"{base}-s{s}" in S}
        sec["seed_floor"] = {}
        for arm in arms:
            for i, j in itertools.combinations(seeds, 2):
                if f"{arm}-s{i}" in S and f"{arm}-s{j}" in S:
                    sec["seed_floor"][f"{arm}: s{i} vs s{j}"] = pair(
                        S[f"{arm}-s{i}"], S[f"{arm}-s{j}"], split)
        sec["native"] = {t: native(S[t]["cif"], n, gt) for t in sorted(tags)}

        # The verdict is a comparison, not a number: the lever's worst per-domain move against
        # the worst move the sampler makes on its own.
        keys = [k for k in ("domain1_all_atom_A", "domain2_all_atom_A", "hinge_free_all_atom_A",
                            "domain1_ca_A", "domain2_ca_A", "lddt_ca", "lddt_ca_domain1",
                            "lddt_ca_domain2", "whole_all_atom_A", "hinge_deg")
                if any(k in v for v in sec["lever"].values())]
        sec["summary"] = {k: {**{arm: spread([v.get(k) for t, v in sec["lever"].items()
                                              if t.startswith(f"{arm} ")]) for arm in arms[1:]},
                              "seed_floor": spread([v.get(k) for v in sec["seed_floor"].values()])}
                          for k in keys}
        for who in arms:
            for metric in ("ca_rmsd_A", "lddt_ca"):
                for dom in GT_SEGMENTS[n]:
                    sec.setdefault("native_summary", {})[f"{who} {dom} {metric}"] = spread(
                        [sec["native"][t][dom][metric] for t in tags
                         if t.split("-")[0] == who and not t.endswith("_r1")])
        report["sizes"][size] = sec

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    for size, sec in report["sizes"].items():
        print(f"\n=== {size} aa ===  A/A bit-exact: {sec.get('aa_floor_bitexact')}")
        for k, v in sec["summary"].items():
            lev = "  ".join(f"{arm} max {v[arm].get('max'):>9}" for arm in arms[1:] if arm in v)
            print(f"  {k:24s} {lev} | seed floor max {v['seed_floor'].get('max'):>9}")
        for k, v in sec.get("native_summary", {}).items():
            print(f"  native {k:28s} {v}")
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
