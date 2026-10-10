"""1HCL CA-lDDT per domain for any number of cdk2x2_512 bench runs, one row per run, seeds listed.

    python perf/spd_wherr/hcl_table.py NAME=RUN_DIR [NAME=RUN_DIR ...] [--out OUT.json]

Each RUN_DIR holds struct_cdk2x2_512_s<seed>/cdk2x2_512.cif (perf/spd/bench.py --out). Same instrument as
perf/spd_overhead/hcl_native.py (perf/b2z2_fusebias/score.py's `native`), without the pairing, so arms on
different architectures or trees sit side by side.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "b2z2_fusebias"))
from score import GT, native  # noqa: E402
from of3_score_ref import ca_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    gt = ca_map(GT)
    rep = {}
    for spec in a.runs:
        name, run = spec.split("=", 1)
        per = {}
        for d in sorted(Path(run).glob("struct_cdk2x2_512_s*")):
            cif = d / "cdk2x2_512.cif"
            if cif.exists():
                per[int(d.name.rsplit("_s", 1)[1])] = native(cif, 512, gt)
        if not per:
            print(f"{name:14s} no folds in {run}"); continue
        doms = list(next(iter(per.values())))
        row = {"seeds": sorted(per)}
        for dom in doms:
            v = np.array([per[s][dom]["lddt_ca"] for s in sorted(per)])
            row[dom] = {"lddt_ca": v.round(4).tolist(), "mean": round(float(v.mean()), 4),
                        "spread": round(float(v.max() - v.min()), 4)}
        rep[name] = row
        print(f"{name:14s} n={len(per)} " + "  ".join(f"{row[d]['mean']:.4f} (spread {row[d]['spread']:.4f})" for d in doms))
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
