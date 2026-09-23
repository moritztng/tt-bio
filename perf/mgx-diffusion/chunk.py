"""Does a chunk boundary change any sample? The same S samples folded at several
--max_parallel_samples widths (plan-chunk.txt, keep=1), compared sample by sample.

    python perf/mgx-diffusion/chunk.py <model> <tokens> <samples>

Structures are written best-first by confidence, so a file's index is a rank, not a sample id,
and a width that changes any confidence reorders them. Each sample is therefore matched to its
nearest sample at the default width (mps unset) by CA RMSD in the shared frame, no superposition:
0.000 means that sample was reproduced exactly. The worst such match over all samples is the
answer; the Kabsch RMSD of the same pair is printed beside it. The unsuffixed copy is rank 0.
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
    # plan-chunk labels are steps-probe-mps-keep (6-1-<w>-1), or 6-1-1 at the default width
    pat = re.compile(rf"out_{re.escape(model)}-{tokens}-{s}-6-1-(?:(\d+)-)?1$")
    runs = {}
    for d in sorted(HERE.glob("work-*/out_*")):
        m = pat.match(d.name)
        got = samples(d) if m else None
        if got:
            runs[m.group(1) or "default"] = got
    if "default" not in runs:
        sys.exit(f"no default-width run kept for {model} {tokens} x {s}: have {sorted(runs)}")
    ref = runs.pop("default")
    print(f"{model} {tokens} tokens x {s} samples, against the default width "
          f"({len(ref)} samples)")
    frame = lambda a, b: float(np.sqrt(((a - b) ** 2).sum(1).mean()))
    for mps, got in sorted(runs.items(), key=lambda kv: int(kv[0])):
        best = [min(((frame(x, y), kabsch_rmsd(x, y), j) for j, y in ref.items()))
                for _i, x in sorted(got.items())]
        moved = [i for (i, _x), b in zip(sorted(got.items()), best) if b[2] != i]
        print(f"  mps={mps:>3}: {len(best)} samples, worst nearest match {max(b[0] for b in best):.3f} A "
              f"in frame ({max(b[1] for b in best):.3f} A Kabsch); {len(moved)} changed rank")


if __name__ == "__main__":
    main()
