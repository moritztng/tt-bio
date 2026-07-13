#!/usr/bin/env python3
"""Repackage an official Protenix 2.0.0 reference dump tree into the harness shape
(`structures/prot.cif` + `results.json`) that scripts/pharma_parity.py structures
expects, one seed dir per invocation.

Protenix dumps to:
  <out_dir>/raw/<dataset>/<pdb>/seed_<seed>/predictions/<pdb>_sample_<i>.cif
  <out_dir>/raw/<dataset>/<pdb>/seed_<seed>/predictions/<pdb>_summary_confidence_sample_<i>.json
The cif files are named by ORIGINAL sample index, not rank position, so this script
reads every summary_confidence json, picks the sample with the max ranking_score
(Protenix's own "best" selection, matching the device leg's confidence-selected
best-of-5), copies that cif to structures/prot.cif, and writes a results.json with
the confidence metadata.

Usage:
  python protenix_ref_to_harness.py <ref_seed_dir> [<target_id>]
e.g.  python protenix_ref_to_harness.py /home/ttuser/pharma_protenix_run/ref_seed0 prot
"""
import glob, json, os, shutil, sys

ref_dir = sys.argv[1]
tid = sys.argv[2] if len(sys.argv) > 2 else "prot"

pred_dir = None
for root, dirs, files in os.walk(os.path.join(ref_dir, "raw")):
    if any(f.endswith("_summary_confidence_sample_0.json") for f in files):
        pred_dir = root
        break
if pred_dir is None:
    raise SystemExit(f"No predictions/ under {ref_dir}/raw; run not complete?")

summaries = glob.glob(os.path.join(pred_dir, "*_summary_confidence_sample_*.json"))
best = None  # (ranking_score, orig_idx, summary_dict)
for sp in summaries:
    with open(sp) as f:
        d = json.load(f)
    rs = d.get("ranking_score", d.get("ptm", 0.0))
    idx = int(sp.rsplit("_sample_", 1)[1].rsplit(".", 1)[0])
    if best is None or rs > best[0]:
        best = (rs, idx, d)
if best is None:
    raise SystemExit("No summary_confidence jsons found in {pred_dir}")
_, best_idx, best_sum = best

cif_src = os.path.join(pred_dir, f"{tid}_sample_{best_idx}.cif")
if not os.path.exists(cif_src):
    # fall back to any *_sample_{best_idx}.cif in pred_dir
    cands = glob.glob(os.path.join(pred_dir, f"*_sample_{best_idx}.cif"))
    if not cands:
        raise SystemExit(f"Best-sample cif not found: {cif_src}")
    cif_src = cands[0]

struct_dir = os.path.join(ref_dir, "structures")
os.makedirs(struct_dir, exist_ok=True)
shutil.copy(cif_src, os.path.join(struct_dir, f"{tid}.cif"))

results = [{
    "id": tid,
    "status": "ok",
    "n_residues": best_sum.get("n_residues"),
    "n_chains": best_sum.get("n_chains", 1),
    "msa": True,
    "samples": len(summaries),
    "ptm": best_sum.get("ptm"),
    "iptm": best_sum.get("iptm"),
    "plddt": best_sum.get("plddt", best_sum.get("mean_plddt")),
    "ranking_score": best_sum.get("ranking_score"),
    "selected_sample_idx": best_idx,
}]
with open(os.path.join(ref_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"REPACK_OK {ref_dir}: structures/{tid}.cif (sample {best_idx}, "
      f"ranking_score={best_sum.get('ranking_score')}, ptm={best_sum.get('ptm')})")
