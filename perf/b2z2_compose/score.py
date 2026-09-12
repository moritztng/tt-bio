#!/usr/bin/env python3
"""Score the composed A/B: ratios, the session's own A/A floor, and the union's additivity discount.

Host only. Reads what `ab_compose.py` wrote and adds the structural check the non-bit-exact arm
owes, at 512 aa AND at 298 aa -- `precision-change-298aa-control-blind-to-512aa-failure` is why
both, not one.

The 512 aa verdict is PER PSEUDO-DOMAIN, not whole-molecule. `cdk2x2_512` is CDK2 followed by its
own residues 1-214 with no interface between the two copies, so the hinge saturates whole-molecule
RMSD for any reassociation: `b2z2-fusebias-512-parity` measured the same arm at four seeds and got
6.80 - 17.36 A whole-molecule while both domains stayed at 0.967 - 1.906 A. A 0.60 A whole-molecule
bar on a fixture with that much hinge noise cannot tell a lever from a re-roll, and it killed two
arms -- HOST at 0.7714 A and the MSA ladder at 0.9765 A -- whose domains read 0.580/0.403 A and
0.709/0.565 A when `b2z2-killed-levers-rescore` split them. The published p150a cell would fail its
own bar by 14x.

So the bar moves onto the reading the fixture supports, and it is the fusebias row's, unchanged:
kill above 0.60 A all-atom on EITHER pseudo-domain, split at `label_seq_id` 298 and each superposed
alone. Whole-molecule RMSD and the hinge angle are still printed, as magnitudes, never as verdicts.
The per-domain arithmetic is imported from `perf/b2z2_fusebias/score.py` rather than re-implemented:
that file reproduces the committed `perf/b2x-integrate/split512.json` numbers to five decimals, and
a second copy of a scorer is a second scorer to keep honest.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
from cif_rmsd import kabsch_rmsd, read_atoms          # noqa: E402

#: The per-domain instrument, loaded under its own module name so this file's own `score` does
#: not shadow it. `SPLIT` is the last `label_seq_id` of pseudo-domain 1 in `cdk2x2_512`.
_FB = importlib.util.spec_from_file_location(
    "b2z2_fusebias_score", ROOT / "perf" / "b2z2_fusebias" / "score.py")
_fb = importlib.util.module_from_spec(_FB)
_FB.loader.exec_module(_fb)
SPLIT = 298

LIVE_CELL_S = 20.079      # site/data/perf-512aa.json, models[0].cells.p150a.s_per_fold @ fc7fed56
PUBLISHED_ALT_S = 23.841  # the denominator wave 1 quoted; kept so both readings are visible
KILL_A = 0.60


def rmsd(pa: Path, pb: Path) -> dict:
    """`pb` (the arm) read against `pa` (the base), per pseudo-domain where there are two.

    Below the split the fixture is one domain and the whole-molecule number IS the verdict; at
    512 aa it is not, and the two domains are.
    """
    a, b = _fb.load(pa, SPLIT), _fb.load(pb, SPLIT)
    split = SPLIT if int(a["seq"].max()) > SPLIT else int(a["seq"].max())
    r = _fb.pair(b, a, split)
    doms = [r[k] for k in ("domain1_all_atom_A", "domain2_all_atom_A") if k in r]
    scored = doms or [r["whole_all_atom_A"]]
    out = {"n_matched": len(a["keys"]),
           "allatom_rmsd_A": round(r["whole_all_atom_A"], 4),
           "ca_rmsd_A": round(r["whole_ca_A"], 4),
           "per_domain_allatom_A": [round(x, 4) for x in doms],
           "worst_domain_allatom_A": round(max(scored), 4),
           "hinge_deg": r.get("hinge_deg"),
           "lddt_ca_vs_base": r.get("lddt_ca"),
           "scored_on": "per_pseudo_domain" if doms else "whole_molecule",
           "verdict": "kill" if max(scored) > KILL_A else "pass",
           "identical_bytes": pa.read_bytes() == pb.read_bytes()}
    return out


def one_cif(d: Path) -> Path:
    c = sorted(d.glob("*.cif"))
    assert c, f"no CIF under {d}"
    return c[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--singles", default=None,
                    help="comma-separated single-lever arms the union is the composition of. "
                         "Defaults to every non-base, non-UNION* arm in the run.")
    ap.add_argument("--singles-union", default=None,
                    help="which UNION* arm the discount is computed against")
    a = ap.parse_args()
    d = json.loads(a.run.read_text())
    timed = [r for r in d["phase1"] if not r["warmup"]]
    seen = list(dict.fromkeys(r["arm"] for r in timed))
    arms = [a for a in seen]
    union = a.singles_union or next((a_ for a_ in arms if a_.startswith("UNION")), None)
    singles = [x.strip() for x in a.singles.split(",")] if a.singles else \
        [x for x in arms if x != "base" and not x.startswith("UNION")]

    per = {}
    for arm in arms:
        v = sorted(r["fold_s"] for r in timed if r["arm"] == arm)
        per[arm] = {"n": len(v), "median_s": round(st.median(v), 3),
                    "min_s": v[0], "max_s": v[-1],
                    "spread_pct": round(100 * (v[-1] - v[0]) / st.median(v), 2),
                    "all_s": v}
    base = per["base"]["median_s"]

    # A/A floor: the two base positions inside each rep, worse over better.
    aa = []
    for i in sorted({r["rep"] for r in timed}):
        b = [r["fold_s"] for r in timed if r["arm"] == "base" and r["rep"] == i]
        if len(b) == 2:
            aa.append(max(b) / min(b))
    floor = round(st.median(aa), 5) if aa else None

    others = [x for x in arms if x != "base"]
    ratio = {arm: round(base / per[arm]["median_s"], 5) for arm in others}
    # An arm's WORST rep against the base median: the pessimistic reading of the same data.
    worst = {arm: round(base / per[arm]["max_s"], 5) for arm in others}
    product = 1.0
    for arm in singles:
        product *= ratio[arm]
    discount_pct = (round(100 * (1 - ratio[union] / product), 3)
                    if union and union in ratio else None)

    # Movement: seconds, not just ratios, against the base arm measured in this same session.
    delta_s = {arm: round(base - per[arm]["median_s"], 3) for arm in others}
    sum_singles_s = round(sum(delta_s[arm] for arm in singles), 3)

    trunk = {arm: round(st.median([r["prepare_and_trunk_s"] for r in timed if r["arm"] == arm]), 4)
             for arm in arms}
    sampler = {arm: round(st.median([r["stages_s"]["sampler"] for r in timed
                                     if r["arm"] == arm and "sampler" in r["stages_s"]]), 4)
               for arm in arms}

    sha = {arm: sorted({r["cif_sha256"] for r in timed if r["arm"] == arm}) for arm in arms}
    bitexact = {arm: (len(sha[arm]) == 1 and sha[arm] == sha["base"]) for arm in arms}

    res = {
        "source": str(a.run), "env": d["env"],
        "per_arm": per, "trunk_median_s": trunk, "sampler_median_s": sampler,
        "fold_AA_ratio": floor, "AA_pairs": len(aa),
        "ratio_vs_base": ratio, "worst_rep_ratio_vs_base": worst,
        "delta_s_vs_base": delta_s, "sum_of_singles_s": sum_singles_s,
        "union_arm": union, "union_singles": singles,
        "union_product_of_singles": round(product, 5),
        "union_additivity_discount_pct": discount_pct,
        "cif_sha256_by_arm": sha, "bit_exact_vs_base": bitexact,
        "vs_live_cell": {arm: round(LIVE_CELL_S / per[arm]["median_s"], 5) for arm in arms},
        "vs_published_23841": {arm: round(PUBLISHED_ALT_S / per[arm]["median_s"], 5) for arm in arms},
        "live_cell_s": LIVE_CELL_S,
    }

    # ---- structure, for the arms that are not bit-exact ---------------------
    struct = {}
    for size in ("512", "298"):
        ref_dirs = sorted(a.cifdir.glob(f"{size}_base_*"))
        if not ref_dirs:
            continue
        ref = one_cif(ref_dirs[0])
        for arm in others:
            ds = sorted(a.cifdir.glob(f"{size}_{arm}_*"))
            if not ds:
                continue
            struct[f"{size}_{arm}"] = rmsd(ref, one_cif(ds[0]))
        if len(ref_dirs) > 1:      # base against its own repeat: the structural A/A floor
            struct[f"{size}_base_repeat"] = rmsd(ref, one_cif(ref_dirs[-1]))
    res["structure_vs_base"] = struct
    if "phase2_298" in d:
        res["phase2_298_fold_s"] = {r["arm"]: r["fold_s"] for r in d["phase2_298"]}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in
                      ("per_arm", "fold_AA_ratio", "ratio_vs_base", "worst_rep_ratio_vs_base",
                       "delta_s_vs_base", "sum_of_singles_s", "union_product_of_singles",
                       "union_additivity_discount_pct", "bit_exact_vs_base", "vs_live_cell",
                       "trunk_median_s", "sampler_median_s", "structure_vs_base")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
