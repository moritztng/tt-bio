"""Does a chunk boundary change any sample? The same S samples folded at several
--max_parallel_samples widths (plan-chunk.txt, keep=1), compared sample by sample.

    python perf/mgx-diffusion/chunk.py <model> <tokens> <samples>

For every kept width, each sample's CA coordinates against the default width's (mps unset):
the RMSD in the shared frame with no superposition (0.000 when the sample is the same sample),
and after Kabsch (how far apart the structures are when the frame is taken out). Files are
matched by name; the ranked copy without a _model_ suffix is skipped.
"""
import re
import sys
from pathlib import Path

import gemmi
import numpy as np

HERE = Path(__file__).resolve().parent


def ca(path):
    st = gemmi.read_structure(str(path))
    return np.array([a.pos.tolist() for r in st[0].all() for a in r if a.name == "CA"])


def kabsch_rmsd(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, _s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


def samples(out):
    return {int(re.search(r"_model_(\d+)\.", f.name).group(1)): ca(f)
            for f in out.rglob("*_model_*.cif")}


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
    for mps, got in sorted(runs.items(), key=lambda kv: int(kv[0])):
        raw = [float(np.sqrt(((got[i] - ref[i]) ** 2).sum(1).mean())) for i in sorted(ref) if i in got]
        fit = [kabsch_rmsd(got[i], ref[i]) for i in sorted(ref) if i in got]
        print(f"  mps={mps:>3}: {len(raw)} samples  frame RMSD max {max(raw):.3f} A "
              f"mean {np.mean(raw):.3f} A | Kabsch max {max(fit):.3f} A mean {np.mean(fit):.3f} A")


if __name__ == "__main__":
    main()
