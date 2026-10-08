#!/usr/bin/env python3
"""Copy tt-bio's Protenix-v2 outputs into the GPU run layout.

    python tt_relayout.py <tt out dir> "<seeds>"

tt-bio writes one results.json per fold whose `all_runs` lists the samples by rank, and the structures as
<id>.cif (rank 0) and <id>_model_<k>.cif (rank k). Sample k here is rank k. `confidence_score` (0.8 ipTM +
0.2 pTM) becomes `ranking_score`, so score.py picks the top-ranked pose by it; pLDDT is rescaled to 0-100.
"""
import json, shutil, sys
from pathlib import Path

root, seeds = Path(sys.argv[1]), "-".join(sys.argv[2].split())
for mode_dir in sorted(root.glob("tt_*")):
    if "_" in mode_dir.name[3:]:
        continue
    for res in sorted(mode_dir.glob("*/seed_*/protenix_results_*/results.json")):
        seed_dir = res.parent.parent
        pdb = seed_dir.parent.name
        dst = root / f"tt{mode_dir.name[3:]}_{seeds}" / "pred" / pdb / seed_dir.name / "predictions"
        dst.mkdir(parents=True, exist_ok=True)
        for run in json.loads(res.read_text())[0]["all_runs"]:
            k = run["rank"]
            conf = dict(run, ranking_score=run["confidence_score"], plddt=100 * run["plddt"])
            shutil.copy(res.parent / "structures" / (f"{pdb}.cif" if k == 0 else f"{pdb}_model_{k}.cif"),
                        dst / f"{pdb}_sample_{k}.cif")
            (dst / f"{pdb}_summary_confidence_sample_{k}.json").write_text(json.dumps(conf))
