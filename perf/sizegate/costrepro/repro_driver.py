"""N back-to-back census folds of protenix-v1 at 256 aa, same card, same commit.

Each _run_census_fold spawns a FRESH process (lever_census.py -> python -m tt_bio.main
predict), so every rep here is already a cold-process rep. runtime_s is the fold's own
timer out of results.json (excludes model load + startup); wall is the whole subprocess.
"""
import importlib.util, json, os, sys, time
from pathlib import Path

REPS = int(os.environ.get("REPRO_REPS", "6"))
MODEL = os.environ.get("REPRO_MODEL", "protenix-v1")
RUNG = int(os.environ.get("REPRO_RUNG", "256"))
root = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("release_gate", root / "scripts" / "release_gate.py")
rg = importlib.util.module_from_spec(spec)
sys.modules["release_gate"] = rg
spec.loader.exec_module(rg)

wd = root / "perf" / "reprowork"
out = []
for i in range(REPS):
    la = os.getloadavg()[0]
    t0 = time.time()
    r = rg._run_census_fold(MODEL, RUNG, wd, f"repro{i}")
    rec = {"rep": i, "loadavg_before": round(la, 2),
           "runtime_s": r.get("runtime_s"), "wall": round(r.get("wall", time.time() - t0), 2),
           "error": r.get("error")}
    out.append(rec)
    print(json.dumps(rec), flush=True)
    (root / "perf" / "reprowork" / "reps.json").write_text(json.dumps(out, indent=2))
ok = [r["runtime_s"] for r in out if r["runtime_s"] is not None]
if ok:
    ok_s = sorted(ok)
    print(json.dumps({"n": len(ok), "min": min(ok), "median": ok_s[len(ok_s)//2],
                      "max": max(ok), "spread_pct": round(100*(max(ok)-min(ok))/min(ok), 1)}), flush=True)
