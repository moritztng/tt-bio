#!/usr/bin/env python3
"""How wide is tt-bio's Protenix ensemble, and does its ranker pick the right sample?

PXDesign's `extended` filter accepts on `ptx_pred_design_rmsd < 2.5 A`, so it is only a filter
if the Protenix fold it measures is that reproducible. This scores the ensemble the shipped
`tt-bio predict` path produces against the committed reference structure for the
`protenix-prot-msa` parity leg -- the gate's own leg, its own staged 157-row MSA, its own
settings (200 steps, 5 samples, 10 cycles, bf16).

    # stage the leg's MSA under the hash name prepare_features looks up, then:
    python3 -m tt_bio.main predict examples/prot.yaml --model protenix-v2 \\
        --out_dir /tmp/prot_prod --msa_dir <staged> --sampling_steps 200 --diffusion_samples 5
    python3 scripts/pxdesign_port/ensemble_probe.py --results /tmp/prot_prod/protenix_results_prot

Reports the inter-sample spread, each sample against the reference, and which sample the
model's own `confidence_score` ranks first. A nearest-match statistic over the two ensembles
cannot answer the last one, which is why it is reported separately.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import gemmi
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preset_run import kabsch_rmsd  # noqa: E402

REF = (REPO / "docs/implementation-parity-data/ref-fixtures/protenix-v2/prot"
              "/msa-server_200step_5sample_10cycle_bf16/seed0/structures/prot.cif")


def ca(path) -> np.ndarray:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_hydrogens()
    return np.array([[a.pos.x, a.pos.y, a.pos.z]
                     for ch in st[0] for r in ch for a in r if a.name == "CA"])


def rg(x) -> float:
    return float(np.sqrt(((x - x.mean(0)) ** 2).sum(-1).mean()))


ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True, help="a protenix_results_<id> directory")
ap.add_argument("--reference", default=str(REF))
ap.add_argument("--out", default=None)
args = ap.parse_args()

d = Path(args.results)
runs = json.loads((d / "results.json").read_text())[0]["all_runs"]
# rank 0 is written as <id>.cif and the rest as <id>_model_<k>.cif
stem = d.name.replace("protenix_results_", "")
paths = [d / "structures" / stem / ".cif"] if False else \
        [d / "structures" / f"{stem}.cif"] + \
        [d / "structures" / f"{stem}_model_{k}.cif" for k in range(1, len(runs))]
dev = [ca(p) for p in paths]
ref = ca(args.reference)

pw = [kabsch_rmsd(dev[i], dev[j]) for i, j in itertools.combinations(range(len(dev)), 2)]
vs = [kabsch_rmsd(ref, x) for x in dev]
rec = {
    "results": str(d), "reference": args.reference, "n_samples": len(dev),
    "inter_sample_rmsd": {"min": round(min(pw), 2), "max": round(max(pw), 2),
                          "mean": round(float(np.mean(pw)), 2)},
    "vs_reference_rmsd": [round(v, 2) for v in vs],
    "confidence_score": [r["confidence_score"] for r in runs],
    "plddt": [r["plddt"] for r in runs],
    "rank0_vs_reference": round(vs[0], 2),
    "best_available_vs_reference": round(min(vs), 2),
    "best_available_rank": int(np.argmin(vs)),
    "rg_reference": round(rg(ref), 2), "rg_device": [round(rg(x), 2) for x in dev],
}
print(json.dumps(rec, indent=1))
print(f"\nthe ranker picked rank 0 at {rec['rank0_vs_reference']} A; "
      f"rank {rec['best_available_rank']} was available at "
      f"{rec['best_available_vs_reference']} A")
if args.out:
    Path(args.out).write_text(json.dumps(rec, indent=1))
