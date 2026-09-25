#!/usr/bin/env python3
"""A25 for the trunk gradient: are we wrong in the same DIRECTION as upstream's own bf16 step?

Two distances from one reference do not order each other (D72), so the headline is followed by
the cosine between the two ERROR vectors -- ours minus float64, and upstream's own bf16 recipe
minus float64. A high cosine says the same mechanism, amplified; a low one says a different
function. Reported over the concatenated set and per leaf op, beside the ratio ours/floor, and
beside the headline restricted to the leaves that actually carry the reference mass -- because a
mass-weighted rel_l2 is driven by the largest ERROR wherever it sits, and on this scope that is
1 % of the reference mass.
"""
import argparse
import json
import math
import re
from pathlib import Path

import torch

_BLK = re.compile(r"^pairformer_stack\.blocks\.(\d+)\.(.*)$")


def leaf_of(n):
    m = _BLK.match(n)
    r = m.group(2) if m else n
    if r.startswith("pair_stack."):
        r = r[len("pair_stack."):]
    return r.split(".")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--identity", action="append", default=[], metavar="A=B",
                    help="assert two dumps are bit-identical")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ld = lambda p: {k: v.double() for k, v in
                    torch.load(p, map_location="cpu", weights_only=False).items() if v is not None}
    ref, floor = ld(a.reference), ld(a.floor)
    arms = {}
    for s in a.arm:
        n, _, p = s.partition("=")
        arms[n] = ld(p)

    keys = set(ref) & set(floor)
    for v in arms.values():
        keys &= set(v)
    keys = sorted(k for k in keys if all(tuple(d[k].shape) == tuple(ref[k].shape)
                                         for d in list(arms.values()) + [floor]))

    out = {"what": __doc__.strip().splitlines()[0], "reference": a.reference, "floor": a.floor,
           "n_keys": len(keys), "identity_controls": [], "arms": {}}

    for spec in a.identity:
        x, _, y = spec.partition("=")
        dx, dy = ld(x), ld(y)
        shared = sorted(set(dx) & set(dy))
        bad = [k for k in shared if not torch.equal(dx[k], dy[k])]
        out["identity_controls"].append(
            {"a": x, "b": y, "keys_a": len(dx), "keys_b": len(dy), "shared": len(shared),
             "n_not_bit_identical": len(bad), "worst": bad[:4],
             "verdict": "PASS -- bit identical" if not bad and len(dx) == len(dy) else "FAIL"})
        print(f"identity {Path(x).name} vs {Path(y).name}: "
              f"{out['identity_controls'][-1]['verdict']} "
              f"({len(shared)} shared, {len(bad)} differing)", flush=True)

    # per-leaf reference mass, so the "leaves that carry the mass" split is measured
    leaf_ref_sq, tot_ref_sq = {}, 0.0
    for k in keys:
        s = float((ref[k] ** 2).sum())
        leaf_ref_sq[leaf_of(k)] = leaf_ref_sq.get(leaf_of(k), 0.0) + s
        tot_ref_sq += s
    MASS_LEAVES = [l for l, s in leaf_ref_sq.items() if s / tot_ref_sq >= 0.02]
    out["leaf_reference_mass_share"] = {l: s / tot_ref_sq for l, s in
                                        sorted(leaf_ref_sq.items(), key=lambda kv: -kv[1])}
    out["mass_carrying_leaves"] = sorted(MASS_LEAVES)
    out["mass_carrying_leaves_rule"] = ">= 2 % of the compared reference squared gradient norm"
    out["mass_carrying_leaves_share"] = sum(leaf_ref_sq[l] for l in MASS_LEAVES) / tot_ref_sq

    fe = {k: (floor[k] - ref[k]).flatten() for k in keys}
    for name, arm in arms.items():
        acc = {}
        for k in keys:
            d = (arm[k] - ref[k]).flatten()
            f = fe[k]
            l = leaf_of(k)
            e = acc.setdefault(l, [0.0] * 5)
            e[0] += float((d * d).sum())           # our error squared
            e[1] += float((f * f).sum())           # floor error squared
            e[2] += float(torch.dot(d, f))         # error-error dot
            e[3] += float((ref[k] ** 2).sum())     # reference squared
            e[4] += 1
        tot = [sum(v[i] for v in acc.values()) for i in range(5)]
        row = {
            "mass_weighted_rel_l2": math.sqrt(tot[0] / tot[3]),
            "floor_mass_weighted_rel_l2": math.sqrt(tot[1] / tot[3]),
            "ours_over_floor": math.sqrt(tot[0] / tot[1]),
            "error_cosine_vs_floor_error": tot[2] / (math.sqrt(tot[0] * tot[1]) + 1e-300),
            "by_leaf": {}, }
        for l, v in sorted(acc.items(), key=lambda kv: -kv[1][3]):
            row["by_leaf"][l] = {
                "n": int(v[4]), "share_of_reference_mass": v[3] / tot[3],
                "share_of_our_error_mass": v[0] / tot[0],
                "rel_l2": math.sqrt(v[0] / v[3]), "floor_rel_l2": math.sqrt(v[1] / v[3]),
                "ours_over_floor": math.sqrt(v[0] / v[1]),
                "error_cosine_vs_floor_error": v[2] / (math.sqrt(v[0] * v[1]) + 1e-300)}
        mk = [l for l in MASS_LEAVES]
        s0 = sum(acc[l][0] for l in mk); s1 = sum(acc[l][1] for l in mk)
        s2 = sum(acc[l][2] for l in mk); s3 = sum(acc[l][3] for l in mk)
        row["mass_carrying_leaves_only"] = {
            "leaves": sorted(mk), "share_of_reference_mass": s3 / tot[3],
            "rel_l2": math.sqrt(s0 / s3), "floor_rel_l2": math.sqrt(s1 / s3),
            "ours_over_floor": math.sqrt(s0 / s1),
            "error_cosine_vs_floor_error": s2 / (math.sqrt(s0 * s1) + 1e-300)}
        out["arms"][name] = row
        print(f"\n{name}: mass-weighted {row['mass_weighted_rel_l2']:.6e}  floor "
              f"{row['floor_mass_weighted_rel_l2']:.6e}  ours/floor {row['ours_over_floor']:.4f}  "
              f"error-cos vs floor {row['error_cosine_vs_floor_error']:.6f}")
        m = row["mass_carrying_leaves_only"]
        print(f"  over the {m['share_of_reference_mass']*100:.2f} % of reference mass in "
              f"{', '.join(m['leaves'])}: {m['rel_l2']:.6e}  floor {m['floor_rel_l2']:.6e}  "
              f"ours/floor {m['ours_over_floor']:.4f}  error-cos {m['error_cosine_vs_floor_error']:.6f}")
        for l, v in row["by_leaf"].items():
            print(f"    {l:20s} ref {v['share_of_reference_mass']*100:6.2f}%  err "
                  f"{v['share_of_our_error_mass']*100:6.2f}%  rel {v['rel_l2']:.3e}  floor "
                  f"{v['floor_rel_l2']:.3e}  o/f {v['ours_over_floor']:8.3f}  "
                  f"err-cos {v['error_cosine_vs_floor_error']:+.4f}")

    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
