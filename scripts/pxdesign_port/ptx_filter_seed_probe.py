"""Does a v0.5.0 Protenix filter checkpoint fold the same target twice?

`extended`'s accept/reject rests on `ptx_pred_design_rmsd < 2.5 A`, so the filter is only a
filter if the fold it measures is reproducible to well inside that. Nothing had ever checked
it: `build_variants_probe.py` proved the three PXDesign-pinned checkpoints BUILD in tt-bio's
`Protenix`, and every parity test that class has is against protenix-v2 weights.

Folds the bare PD-L1 target (116 residues, its cached MSA) at two seeds and reports the CA
RMSD of each against the crystal, the same RMSD allowing a reflection, and the two folds
against each other.

    TT_VISIBLE_DEVICES=0 python3 scripts/pxdesign_port/ptx_filter_seed_probe.py [--n_step 200]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pxdesign_port"))

from preset_run import ca_rows, kabsch_rmsd, load_ptx, ptx_features, ptx_fold, read_a3m  # noqa: E402
from tt_bio.protenix_data import structure_token_coords  # noqa: E402
from tt_bio.pxdesign.inputs import read_design_yaml  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--yaml", default=str(REPO / "tests" / "fixtures" / "pxdesign" / "PDL1.yaml"))
ap.add_argument("--msa", default=str(Path.home() / "pxdesign_src" / "examples" / "msa" / "PDL1" / "0"))
ap.add_argument("--variant", default="base")
ap.add_argument("--n_step", type=int, default=200)
ap.add_argument("--seeds", default="0,1")
args = ap.parse_args()

spec = read_design_yaml(args.yaml)
chain = structure_token_coords(spec["structure"], spec["chains"], spec["crop"])[spec["chains"][0]]
ca = chain["ca"].numpy()
model, witness = load_ptx(args.variant)
feats = ptx_features([(chain["sequence"], read_a3m(args.msa), "protein")])
print(f"{witness['checkpoint_name']}  {witness['params_m']} M params  "
      f"{len(ca)} residues  n_step={args.n_step}", flush=True)

MIRROR = np.array([-1.0, 1.0, 1.0])
folds = []
for seed in [int(x) for x in args.seeds.split(",")]:
    coords, conf = ptx_fold(model, feats, seed, args.variant, n_step=args.n_step)
    pred = coords[ca_rows(feats)].numpy()
    folds.append(pred)
    print(f"  seed {seed}: crystal RMSD {kabsch_rmsd(ca, pred):6.2f} A   "
          f"mirrored {kabsch_rmsd(ca, pred * MIRROR):6.2f} A   "
          f"Rg {np.sqrt(((pred - pred.mean(0)) ** 2).sum(-1).mean()):5.2f} A   "
          f"pLDDT {conf['plddt']:.3f}  pTM {conf['ptm']:.3f}", flush=True)
for i in range(len(folds) - 1):
    print(f"  fold {i} vs fold {i + 1}: {kabsch_rmsd(folds[i], folds[i + 1]):.2f} A", flush=True)
print(f"  crystal Rg {np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()):.2f} A")
