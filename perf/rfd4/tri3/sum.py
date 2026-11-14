"""tri3 summary: in-trunk paired A/B (ab3) and the 8-seed design grade (design4), as one json.

    python perf/rfd4/tri3/sum.py <runs dir> <out.json>

ab3: per size and arm, base / stack s/design (180 x median traced pass), paired speedup, AICLK during.
design4: per size, median warm design_s / trunk_s per arm with AICLK, then perf/rfd4/bfp8/grade_seeds.py
(CA RMSD of stack and base2 against base, beside the seed floor).
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
W = Path(__file__).resolve().parents[3]
out = {"ab3": {}, "design4": {}}
for f in sorted((R / "ab3").glob("*.json")):
    d = json.loads(f.read_text())
    clk = d["aiclk"]["0"]
    out["ab3"][f.stem] = dict(base_s=d["base"]["trunk_s_per_design"], stack_s=d["stack"]["trunk_s_per_design"],
                              speedup=d["paired_speedup_median"], z_rel_rms=d["z_rel_rms_stack_vs_base"],
                              aiclk_median=clk["median"], aiclk_min=clk["min"], load_end=d["load_end"],
                              reps=d["reps"])
for nd in sorted(p for p in (R / "design4").iterdir() if p.is_dir()):
    res = {}
    for arm in ("base", "stack", "base2"):
        f = nd / arm / "runs.jsonl"
        if not f.exists():
            continue
        warm = [r for r in map(json.loads, filter(str.strip, f.read_text().splitlines())) if r["kind"] == "warm"]
        if not warm:
            continue
        mhz = [r["aiclk_mhz"] for r in warm if r.get("aiclk_mhz")]
        res[arm] = dict(n_warm=len(warm), design_s=statistics.median(r["design_s"] for r in warm),
                        trunk_s=statistics.median(r["trunk_s"] for r in warm if r.get("trunk_s")),
                        aiclk_median=statistics.median(mhz) if mhz else None,
                        clock_artifact=any(r.get("clock_artifact") for r in warm),
                        load1_max=max(r.get("load1", 0) for r in warm))
    arms = [str(nd / a) for a in ("stack", "base2") if a in res]
    if "base" in res and arms:
        g = nd / "grade.json"
        cmd = [sys.executable, str(W / "perf/rfd4/bfp8/grade_seeds.py"), "--base", str(nd / "base"), "--out", str(g)]
        for a in arms:
            cmd += ["--arm", a]
        subprocess.run(cmd, check=False, cwd=W)
        if g.exists():
            res["grade"] = json.loads(g.read_text())
    out["design4"][nd.name] = res
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(out["ab3"], indent=1))
