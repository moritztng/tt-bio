"""Fix vs control: same point, same seed. Per rank: max |dCA| in frame; per fix sample: nearest control sample by Kabsch."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/agent/wt-mgx-diffusion/perf/mgx-diffusion")
from chunk import samples, kabsch_rmsd
a, b = samples(Path(sys.argv[1])), samples(Path(sys.argv[2]))
for k in sorted(a):
    d = np.abs(a[k] - b[k]).max() if k in b else float("nan")
    near = min(kabsch_rmsd(a[k], q) for q in b.values())
    print(f"rank {k}: max|dCA| {d:.4f} A in frame, nearest control sample {near:.3f} A Kabsch")
