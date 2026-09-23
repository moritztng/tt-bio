"""Does a chunk boundary change any sample? The same S samples folded at several
--max_parallel_samples widths (plan-chunk.txt, keep=1), compared sample by sample.

    python perf/mgx-diffusion/chunk.py <model> <tokens> <samples> [steps label, default 6-1]

Structures are written best-first by confidence, so a file's index is a rank, not a sample id,
and a width that changes any confidence reorders them. Each sample is therefore matched to its
nearest sample at the default width (mps unset) by CA RMSD in the shared frame, no superposition:
0.000 means that sample was reproduced exactly. The worst such match over all samples is the
answer; the Kabsch RMSD of the same pair is printed beside it. The unsuffixed copy is rank 0.
The floor is the default width's own spread: each sample's Kabsch RMSD to its nearest sibling,
which is what one more noise draw moves a structure. The card each width ran on is printed,
because a difference between cards is a different finding from a difference between widths.
"""
import re
import sys
from pathlib import Path

import gemmi
import numpy as np

HERE = Path(__file__).resolve().parent


def ca(path):
    st = gemmi.read_structure(str(path))
    return np.array([c.atom.pos.tolist() for c in st[0].all() if c.atom.name == "CA"])


def kabsch_rmsd(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, _s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


def samples(out):
    """Rank -> CA coordinates; the file without a _model_ suffix is rank 0."""
    got = {}
    for f in out.rglob("*.cif"):
        m = re.search(r"_model_(\d+)\.cif$", f.name)
        got[int(m.group(1)) if m else 0] = ca(f)
    return got


def main():
    model, tokens, s = sys.argv[1], sys.argv[2], sys.argv[3]
    head = sys.argv[4] if len(sys.argv) > 4 else "6-1"
    # plan-chunk labels are <steps-probe>-<mps>-<keep>; no mps at the default width. Production
    # points carry no steps or probe, so their label is <mps>-1 or bare 1 (pass head "").
    pre = rf"-{head}" if head else ""
    pat = re.compile(rf"out_{re.escape(model)}-{tokens}-{s}{pre}-(?:(\d+)-)?1$")
    cards = {}
    runs = {}
    for d in sorted(HERE.glob("work-*/out_*")):
        m = pat.match(d.name)
        got = samples(d) if m else None
        if got:
            runs[m.group(1) or "default"] = got
            cards[m.group(1) or "default"] = d.parent.name.split("-")[1]
    if "default" not in runs:
        sys.exit(f"no default-width run kept for {model} {tokens} x {s}: have {sorted(runs)}")
    ref = runs.pop("default")
    sib = [min(kabsch_rmsd(x, y) for j, y in ref.items() if j != i) for i, x in ref.items()]
    print(f"{model} {tokens} tokens x {s} samples, against the default width "
          f"({len(ref)} samples, card {cards['default']}); floor: nearest sibling Kabsch "
          f"min {min(sib):.3f} A median {float(np.median(sib)):.3f} A")
    frame = lambda a, b: float(np.sqrt(((a - b) ** 2).sum(1).mean()))
    for mps, got in sorted(runs.items(), key=lambda kv: int(kv[0])):
        best = [min(((frame(x, y), kabsch_rmsd(x, y), j) for j, y in ref.items()))
                for _i, x in sorted(got.items())]
        moved = [i for (i, _x), b in zip(sorted(got.items()), best) if b[2] != i]
        print(f"  mps={mps:>3}: {len(best)} samples, worst nearest match {max(b[0] for b in best):.3f} A "
              f"in frame ({max(b[1] for b in best):.3f} A Kabsch); {len(moved)} changed rank; "
              f"card {cards[mps]}")


if __name__ == "__main__":
    main()
