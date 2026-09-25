#!/usr/bin/env python3
"""Is a verb's pooled injection growth its ERRORS growing, or its WEIGHTS moving?

`VERBS.json`'s `rel_l2_aggregate` pools a verb's firings as
sqrt(sum_i (rel_i * ref_i)^2 / sum_i ref_i^2), so it is a reference-norm-weighted RMS. Two
different things raise it between widths: every firing getting worse, or the reference-norm mass
moving onto the firings that were already worst. The second is not a width effect in the verb at
all. This separates them by crossing the two widths' error vectors with the two widths' weights.

    sm_reweight.py --side64 SIDE_C64.json --side384 SIDE_C384.json --verb softmax --out X.json
"""
from __future__ import annotations

import argparse
import json


def pull(path, verb):
    rel, mass = {}, {}
    for r in json.load(open(path))["nodes"]:
        if r["verb"] != verb:
            continue
        for p in r["parents"]:
            t = p.get("own")
            if not t or not t.get("ref_norm"):
                continue
            rel[r["sm_ord"]] = t["rel_l2"]
            mass[r["sm_ord"]] = t["ref_norm"] ** 2
    return rel, mass


def pooled(rel, mass):
    keys = set(rel) & set(mass)
    den = sum(mass[k] for k in keys)
    return (sum(rel[k] ** 2 * mass[k] for k in keys) / den) ** 0.5 if den else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side64", required=True)
    ap.add_argument("--side384", required=True)
    ap.add_argument("--verb", default="softmax")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    r64, m64 = pull(a.side64, a.verb)
    r384, m384 = pull(a.side384, a.verb)
    base = pooled(r64, m64)

    cells = {
        "errors_64_weights_64": pooled(r64, m64),
        "errors_384_weights_384": pooled(r384, m384),
        "errors_384_weights_64": pooled(r384, m64),
        "errors_64_weights_384": pooled(r64, m384),
    }
    den64, den384 = sum(m64.values()), sum(m384.values())
    body = {
        "what": __doc__.strip().splitlines()[0],
        "verb": a.verb,
        "firings": {"64": len(r64), "384": len(r384)},
        "pooled": cells,
        "growth_over_the_64_cell": {k: (v / base if base else None) for k, v in cells.items()},
        "weight_share_by_ordinal": {
            "64": {str(o): m64[o] / den64 for o in sorted(m64)},
            "384": {str(o): m384[o] / den384 for o in sorted(m384)},
        },
        "per_firing_ratio": {str(o): (r384[o] / r64[o]) for o in sorted(set(r64) & set(r384))
                             if r64[o]},
    }
    rs = sorted(body["per_firing_ratio"].values())
    body["per_firing_ratio_summary"] = {
        "n": len(rs), "min": rs[0], "median": rs[len(rs) // 2], "max": rs[-1],
        "above_1.0": sum(1 for x in rs if x > 1.0),
    }
    json.dump(body, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in body.items()
                      if k not in ("weight_share_by_ordinal", "per_firing_ratio")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
