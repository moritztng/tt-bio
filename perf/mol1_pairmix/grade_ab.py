"""Grade an e2e_ab.sh run: warm fold seconds + AICLK per arm, and the same-seed all-atom RMSD (Kabsch) between the
arms against the base arm's own seed-to-seed RMSD (the floor a lever has to sit well inside).

Usage: python perf/mol1_pairmix/grade_ab.py OUTDIR [BASE_ARM OTHER_ARM] -> prints and writes OUTDIR/grade.json
"""
import itertools
import json
import statistics
import sys

import torch


def rmsd(a, b):
    a, b = a.double() - a.double().mean(0), b.double() - b.double().mean(0)
    u, _, vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.det(u @ vt))
    r = u @ torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=torch.float64)) @ vt
    return float(((a @ r - b) ** 2).sum(1).mean().sqrt())


def main(out, base="base", other="ws"):
    A, B = (torch.load(f"{out}/{n}.pt") for n in (base, other))
    warm = sorted(set(A["coords"]) & set(B["coords"]))[1:]   # the first seed carries the compile
    res = {"seeds": warm}
    for n, R in ((base, A), (other, B)):
        res[n] = {"s": [R["s"][s] for s in warm], "s_median": statistics.median(R["s"][s] for s in warm),
                  "clock": [R["clock"][s] for s in warm], "plddt": [round(R["plddt"][s], 4) for s in warm]}
    res["same_seed_rmsd_A"] = [round(rmsd(A["coords"][s], B["coords"][s]), 3) for s in warm]
    res["seed_floor_rmsd_A"] = [round(rmsd(A["coords"][s], A["coords"][t]), 3) for s, t in itertools.combinations(warm, 2)]
    res["delta_s_median"] = round(res[other]["s_median"] - res[base]["s_median"], 2)
    json.dump(res, open(f"{out}/grade.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:])
