"""Lever-census A/B of the narrow-q flag on single size-ladder cells, within one tree.

The size-ladder arm refuses a rung subset in check mode, and rightly: it exists to read every
rung. This asks a narrower question it cannot: at a cell the full ladder flagged red, does the
drift move with TT_BIO_TRIATT_NARROW_Q_FALLBACK, or is it main's? Same tree, same fixture, same
fold config as the arm (it calls the arm's own _run_census_fold); only the flag differs.

usage: narrowq_cell_ab.py OUT.json model:rung:arm[,model:rung:arm...]   (arm is on or off)
"""
import json
import os
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(WT))
import release_gate as rg  # noqa: E402

FLAG = "TT_BIO_TRIATT_NARROW_Q_FALLBACK"
out = Path(sys.argv[1])
work = out.parent / "cell_ab_work"
rows = json.loads(out.read_text()) if out.exists() else []
for cell in sys.argv[2].split(","):
    model, rung, arm = cell.split(":")
    os.environ[FLAG] = "1" if arm == "on" else "0"
    r = rg._run_census_fold(model, int(rung), work, f"nq{arm}" + os.environ.get("NQ_TAG", ""), need_runtime=False)
    row = {"model": model, "rung": int(rung), "arm": arm, "error": r.get("error"),
           "levers": r.get("levers"), "runtime_s": r.get("runtime_s"), "aiclk": r.get("aiclk"),
           "structure": r.get("structure")}
    rows.append(row)
    out.write_text(json.dumps(rows, indent=1, default=str))
    print(model, rung, arm, "error" if row["error"] else "ok", row["error"] or row["runtime_s"],
          flush=True)
