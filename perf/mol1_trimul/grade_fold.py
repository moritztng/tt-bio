"""Grade a fold_ab.py A/B: warm seconds + AICLK per arm, and per diffusion sample the all-atom RMSD (Kabsch) and lDDT of
each arm against the base arm at the same seed, beside the base arm's own sample-to-sample spread (the floor).
Needs perf/mol1_pairmix/grade_ab.py (rmsd, lddt), on main once the Mol-1 perf stack lands.

Usage: python perf/mol1_trimul/grade_fold.py DIR SIZE BASE_ARM ARM [ARM ...] -> prints and writes DIR/grade_<SIZE>.json
"""
import itertools
import json
import statistics
import sys

import torch

from perf.mol1_pairmix.grade_ab import lddt, rmsd


def main(d, size, base, *arms):
    def load(a):
        r = json.load(open(f"{d}/s{size}_{a}.json"))
        x = torch.load(f"{d}/s{size}_{a}.pt")["coords"].float()
        return r, x.reshape(-1, x.shape[-2], 3)
    rb, xb = load(base)
    warm = lambda r: [f for f in r["folds"] if f["tag"] == "warm"]  # noqa: E731
    pairs = list(itertools.combinations(range(len(xb)), 2))
    res = {"size": size, "input": rb["input"], "tokens": sum(rb["chain_lens"]),
           "floor_rmsd_A": [round(rmsd(xb[i], xb[j]), 3) for i, j in pairs],
           "floor_lddt": [round(lddt(xb[i], xb[j]), 4) for i, j in pairs]}
    for a in (base, *arms):
        r, x = load(a)
        w = warm(r)
        res[a] = {"warm_s": [f["wall_s"] for f in w], "warm_s_median": statistics.median(f["wall_s"] for f in w),
                  "clock": [f["clock"] for f in w], "t1": w[-1]["t1_served_declined"], "t2": w[-1]["t2"],
                  "plddt": r.get("plddt"), "ptm": r.get("ptm")}
        if a != base:
            res[a]["rmsd_vs_base_A"] = [round(rmsd(xb[i], x[i]), 3) for i in range(len(xb))]
            res[a]["lddt_vs_base"] = [round(lddt(xb[i], x[i]), 4) for i in range(len(xb))]
            res[a]["delta_s"] = round(res[a]["warm_s_median"] - res[base]["warm_s_median"], 2)
    json.dump(res, open(f"{d}/grade_{size}.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:])
