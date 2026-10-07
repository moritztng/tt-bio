#!/usr/bin/env python3
"""Copy tt-bio's outputs (<stem>.cif, <stem>_model_k.cif, *_summary_confidences.json) into the GPU run layout.

    python tt_relayout.py <tt out dir> "<seeds>"

tt-bio ranks its samples before writing (rank 0 = best ranking_score), so sample k here is rank k; score.py
picks the top-ranked pose by ranking_score either way. pLDDT is rescaled to Protenix's 0-100.
"""
import json, shutil, sys
from pathlib import Path

root, seeds = Path(sys.argv[1]), "-".join(sys.argv[2].split())
for mode_dir in sorted(root.glob("tt_*")):
    if "_" in mode_dir.name[3:]:
        continue
    for seed_dir in sorted(mode_dir.glob("*/seed_*")):
        pdb = seed_dir.parent.name
        dst = root / f"tt{mode_dir.name[3:]}_{seeds}" / "pred" / pdb / seed_dir.name / "predictions"
        dst.mkdir(parents=True, exist_ok=True)
        for js in sorted(seed_dir.rglob("*_summary_confidences.json")):
            stem = js.name[: -len("_summary_confidences.json")]
            k = int(stem.rsplit("_model_", 1)[1]) if "_model_" in stem else 0
            conf = json.loads(js.read_text())
            if conf.get("plddt") is not None and conf["plddt"] <= 1:
                conf["plddt"] *= 100
            shutil.copy(js.with_name(f"{stem}.cif"), dst / f"{pdb}_sample_{k}.cif")
            (dst / f"{pdb}_summary_confidence_sample_{k}.json").write_text(json.dumps(conf))
