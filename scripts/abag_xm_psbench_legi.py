"""Phase-2 leg (i): our per-interface DockQ vs PSBench's dockq_wave, on PSBench's AF3 models.

dockq_wave is "the weighted average of DockQ scores across all chain-chain interfaces"
(PSBench's own Quality_Scores_Definitions.json). Ours is ONE declared Ab-Ag interface. So the
falsifiable prediction is: strong positive rank correlation, dockq_wave systematically ABOVE
ours, and the gap WIDENING as our score falls -- because their average includes the near-rigid
intra-antibody H-L interface, which is almost always modelled well and dilutes a bad Ab-Ag one.
A flat or positive gap trend would mean our label is not tracking the interface we think it is.
"""
import json
import os
import pathlib
import subprocess
import sys
import csv

import numpy as np
from scipy.stats import spearmanr, pearsonr

WT = pathlib.Path("/home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p4")
PSB = pathlib.Path("/home/ttuser/abag_xm/psbench")
PM = PSB / "Multimer_7_2024_8_2025_dataset/Predicted_Models"
PY = "/home/ttuser/tt-bio/env/bin/python3"
N_SUB = 50           # matches Tier A's own ensemble size; 50 of 200
SEED = 42            # fixed so the leg is reproducible

TARGETS = json.load(open(WT / "docs/implementation-parity-data/abag-xm-psbench-leg-i-targets.json"))
want = {t["id"]: t for t in TARGETS if (PM / t["id"]).is_dir()}
print("targets with models extracted: %s" % ", ".join(sorted(want)))

allrows = []
for tid, meta in sorted(want.items()):
    wave = {}
    for r in csv.DictReader(open(PSB / "Quality_Scores" / f"{tid}_quality_scores.csv")):
        try:
            wave[r["model_name"]] = float(r["dockq_wave"])
        except (TypeError, ValueError):
            pass
    models = sorted((PM / tid / tid).glob("ranked_*.pdb"))
    rng = np.random.default_rng(SEED)
    pick = [models[i] for i in sorted(rng.choice(len(models), size=min(N_SUB, len(models)),
                                                 replace=False))]
    native = PSB / "native" / f"{tid}.cif"
    print("\n%s  H=%s L=%s A=%s  %s  (%d models, %d sampled)"
          % (tid, meta["H"], meta["L"], meta["A"], meta["antigen"][:40], len(models), len(pick)))
    ok = fail = 0
    for m in pick:
        w = wave.get(m.name)
        if w is None:
            continue
        rec = {"target": tid, "model": m.name, "dockq_wave": w}
        pairs = [("HA", meta["H"])]
        if os.environ.get("ABAG_LEGI_LA"):
            pairs.append(("LA", meta["L"]))
        for label, ab in pairs:
            if not ab:
                continue
            p = subprocess.run([PY, str(WT / "scripts/abag_xm_dockq_interface.py"),
                                str(m), str(native), ab, meta["A"]],
                               capture_output=True, text=True, cwd=str(WT),
                               env={"PYTHONPATH": str(WT), "PATH": "/usr/bin:/bin"})
            try:
                out = json.loads(p.stdout)
                rec[label] = out.get("DockQ", out.get("dockq"))
                rec[label + "_status"] = out.get("status", "ok")
            except Exception:
                rec[label] = None
                rec[label + "_err"] = (p.stderr or p.stdout or "")[-160:]
        if rec.get("HA") is None:
            fail += 1
            if fail == 1:
                print("   first failure: %s" % rec.get("HA_err", "?"))
        else:
            ok += 1
        allrows.append(rec)
    print("   scored %d, failed %d" % (ok, fail))

json.dump(allrows, open(PSB / "leg_i_pilot.json", "w"), indent=1)

print("\n================ LEG (i) RESULT ================")
pooled_ours, pooled_wave = [], []
for tid in sorted(want):
    rs = [r for r in allrows if r["target"] == tid and r.get("HA") is not None]
    if len(rs) < 8:
        print("%s: only %d scored -- skipping statistics" % (tid, len(rs)))
        continue
    ours = [r["HA"] for r in rs]
    wv = [r["dockq_wave"] for r in rs]
    pooled_ours += ours
    pooled_wave += wv
    gap = [w - o for w, o in zip(wv, ours)]
    print("%s n=%2d  spearman=%+.3f  mean(wave-ours)=%+.3f  corr(gap, ours)=%+.3f"
          % (tid, len(rs), spearmanr(ours, wv).correlation, float(np.mean(gap)),
             pearsonr(gap, ours)[0]))
if len(pooled_ours) >= 8:
    gap = [w - o for w, o in zip(pooled_wave, pooled_ours)]
    print("\nPOOLED n=%d" % len(pooled_ours))
    print("  spearman(our per-interface DockQ, dockq_wave) = %+.3f" % spearmanr(pooled_ours, pooled_wave).correlation)
    print("  mean(dockq_wave - ours)                       = %+.3f  (expect > 0)" % float(np.mean(gap)))
    print("  corr(gap, ours)                               = %+.3f  (expect < 0)" % pearsonr(gap, pooled_ours)[0])
    print("  our DockQ  min %.3f median %.3f max %.3f" % (min(pooled_ours), float(np.median(pooled_ours)), max(pooled_ours)))
    print("  dockq_wave min %.3f median %.3f max %.3f" % (min(pooled_wave), float(np.median(pooled_wave)), max(pooled_wave)))
