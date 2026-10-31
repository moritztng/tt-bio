"""Grade two run.py arms of one input against each other: warm s/prediction and AICLK per arm, and per diffusion sample
the all-atom RMSD (Kabsch) and lDDT of the arm against the base arm at the same seed, beside the base arm's own
seed-to-seed RMSD over the same samples (the floor a lever has to sit well inside) and the 0.60 A bar.

Usage: python perf/mol1/fair/grade_ab.py BASE_DIR ARM_DIR [--out grade.json]
Each DIR is a one-chip run.py --out (runs.jsonl + chip*/r<i>-s<seed>/ CIFs); arms pair by seed.
"""
import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from perf.mol1_pairmix.grade_ab import lddt, rmsd  # noqa: E402

BAR_A = 0.60


def cif_xyz(path):
    """All-atom coordinates of an mmCIF's _atom_site loop, in file order."""
    cols, xyz, in_loop = [], [], False
    for line in Path(path).read_text().splitlines():
        if line.startswith("_atom_site."):
            cols.append(line.split()[0][len("_atom_site."):])
            in_loop = True
        elif in_loop and cols and line.startswith(("ATOM", "HETATM")):
            f = line.split()
            xyz.append([float(f[cols.index(k)]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")])
        elif in_loop and xyz and (line.startswith("#") or line.startswith("loop_")):
            break
    return torch.tensor(xyz, dtype=torch.float64)


def rounds(d):
    """seed -> {wall_s, clock, samples: [xyz, ...]} for the clean warm rounds of one run.py out dir."""
    out = {}
    for line in open(Path(d) / "runs.jsonl"):
        r = json.loads(line)
        if r.get("kind") != "warm" or r.get("discard"):
            continue
        c = r["per_chip"][0]
        if c.get("err") or not c.get("finite"):
            continue
        cifs = sorted(Path(c["struct_dir"]).rglob("*.cif"))
        out[c["seed"]] = {"wall_s": r["wall_s"], "clock": r.get("aiclk_mhz"), "samples": [cif_xyz(f) for f in cifs]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("arm")
    ap.add_argument("--out")
    a = ap.parse_args()
    b, x = rounds(a.base), rounds(a.arm)
    same = sorted(set(b) & set(x))
    dev = [(rmsd(p, q), lddt(p, q)) for s in same for p, q in zip(b[s]["samples"], x[s]["samples"])]
    floor = [rmsd(p, q) for s, t in itertools.combinations(sorted(b), 2) for p, q in zip(b[s]["samples"], b[t]["samples"])]
    med = lambda v: round(statistics.median(v), 3) if v else None  # noqa: E731
    res = {"base": a.base, "arm": a.arm, "seeds": same,
           "warm_s": {"base": [b[s]["wall_s"] for s in sorted(b)], "arm": [x[s]["wall_s"] for s in sorted(x)]},
           "warm_s_median": {"base": med([b[s]["wall_s"] for s in b]), "arm": med([x[s]["wall_s"] for s in x])},
           "clock": {"base": [b[s]["clock"] for s in sorted(b)], "arm": [x[s]["clock"] for s in sorted(x)]},
           "same_seed_rmsd_A": {"median": med([d[0] for d in dev]), "max": round(max(d[0] for d in dev), 3) if dev else None,
                                "n": len(dev)},
           "same_seed_lddt_median": med([d[1] for d in dev]),
           "seed_floor_rmsd_A": {"median": med(floor), "n": len(floor)}, "bar_A": BAR_A}
    m = res["same_seed_rmsd_A"]["median"]
    res["pass"] = bool(m is not None and m <= BAR_A)
    print(json.dumps(res, indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
