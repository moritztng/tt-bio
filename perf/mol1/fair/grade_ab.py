"""Grade two run.py arms of one input against each other: warm s/prediction and AICLK per arm, and per diffusion sample
the all-atom RMSD (Kabsch) and lDDT of the arm against the base arm at the same seed, beside the base arm's own
spread (the floor a lever has to sit well inside) and the 0.60 A bar. The floor pairs the base arm's distinct folds:
sample-to-sample within a fold, which is independent diffusion noise, and fold-to-fold across seeds. A JSON input's own
seed overrides run.py's per-round seed, so its rounds repeat one fold; repeated folds are counted once.
The CIFs of a fold are ranked by ranking_score, not numbered by noise draw, so a lever that reorders the ranking
would pair different draws by file name. Samples pair by the assignment with the least total RMSD instead; the pairing
is unambiguous when every matched pair sits below the closest two distinct base samples (`pairing_unambiguous`).

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
        sd = Path(c["struct_dir"])
        if not sd.exists():  # a run dir copied off its host: chip<N>/r<i>-s<seed> under d
            sd = Path(d) / sd.parent.name / sd.name
        cifs = sorted(sd.rglob("*.cif"))
        out[c["seed"]] = {"wall_s": r["wall_s"], "clock": r.get("aiclk_mhz"), "samples": [cif_xyz(f) for f in cifs]}
    return out


def matched(base, arm):
    """(base, arm) sample pairs under the assignment with the least total RMSD."""
    d = [[rmsd(p, q) for q in arm] for p in base]
    best = min(itertools.permutations(range(len(arm))), key=lambda pi: sum(d[i][j] for i, j in enumerate(pi)))
    return [(base[i], arm[j]) for i, j in enumerate(best)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("arm")
    ap.add_argument("--out")
    a = ap.parse_args()
    b, x = rounds(a.base), rounds(a.arm)
    same = sorted(set(b) & set(x))
    folds = {}  # distinct base folds, keyed by their coordinates
    for s in sorted(b):
        folds.setdefault(tuple(hash(c.numpy().tobytes()) for c in b[s]["samples"]), b[s]["samples"])
    pairs = {(int(hash(p.numpy().tobytes())), int(hash(q.numpy().tobytes()))): (p, q) for p, q in itertools.combinations(
        [c for f in folds.values() for c in f], 2) if not torch.equal(p, q)}
    dev = [(rmsd(p, q), lddt(p, q)) for s in same for p, q in matched(b[s]["samples"], x[s]["samples"])]
    floor = [(rmsd(p, q), lddt(p, q)) for p, q in pairs.values()]
    med = lambda v: round(statistics.median(v), 3) if v else None  # noqa: E731
    res = {"base": a.base, "arm": a.arm, "seeds": same,
           "warm_s": {"base": [b[s]["wall_s"] for s in sorted(b)], "arm": [x[s]["wall_s"] for s in sorted(x)]},
           "warm_s_median": {"base": med([b[s]["wall_s"] for s in b]), "arm": med([x[s]["wall_s"] for s in x])},
           "clock": {"base": [b[s]["clock"] for s in sorted(b)], "arm": [x[s]["clock"] for s in sorted(x)]},
           "same_seed_rmsd_A": {"median": med([d[0] for d in dev]), "max": round(max(d[0] for d in dev), 3) if dev else None,
                                "n": len(dev)},
           "same_seed_lddt_median": med([d[1] for d in dev]),
           "distinct_base_folds": len(folds),
           "seed_floor_rmsd_A": {"median": med([f[0] for f in floor]), "min": round(min(f[0] for f in floor), 3) if floor else None,
                                 "n": len(floor)},
           "seed_floor_lddt_median": med([f[1] for f in floor]), "bar_A": BAR_A}
    m = res["same_seed_rmsd_A"]["median"]
    res["pairing_unambiguous"] = bool(dev and floor and max(d[0] for d in dev) < min(f[0] for f in floor))
    res["pass"] = bool(m is not None and m <= BAR_A)
    print(json.dumps(res, indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
