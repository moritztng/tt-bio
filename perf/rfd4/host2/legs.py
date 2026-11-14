"""Warm medians per host_split leg: python3 legs.py <leg dir>..."""
import json, statistics as st, sys
from pathlib import Path
K = ("design_s", "wall_s", "device_s", "prep_s", "glue_s", "write_s")
for d in sys.argv[1:]:
    w = [json.loads(l) for l in open(Path(d) / "split.jsonl")]
    w = [r for r in w if r["kind"] == "warm"]
    med = {k: round(st.median(r[k] for r in w), 3) for k in K}
    clk = [r["aiclk_mhz"] for r in w]
    print(Path(d).name, med, "aiclk", clk, "load1", [r["load1"] for r in w],
          "artifact" if any(r["clock_artifact"] for r in w) else "clean")
