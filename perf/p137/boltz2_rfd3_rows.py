"""Does registering RFD3s two model-scoped fusion levers disturb another models census?

Section 24.6 item 1. The size-ladder ARM cannot answer this on qb2: every card here is p300c and
docs/size_ladder_baseline.json only holds p150a, so the arm fails at the baseline lookup before it
folds anything. The question underneath the arm is card-independent, so ask it directly, through
the arms own fold path: census one boltz-2 fold and read the two RFD3 rows.
"""
import importlib.util, json, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rg", REPO / "scripts" / "release_gate.py")
rg = importlib.util.module_from_spec(spec); sys.modules["rg"] = rg; spec.loader.exec_module(rg)

wd = REPO / "perf" / "p137" / "boltz2_census_wd"
r = rg._run_census_fold("boltz2", 256, wd, "rfd3rows")
if r.get("error"):
    print("FOLD ERROR:", r["error"]); sys.exit(2)

levers = r["levers"]
rfd3 = {f: v for f, v in levers.items() if f.startswith("RFD3_")}
print("boltz2 256aa fold, runtime_s", r.get("runtime_s"), "grid", r.get("grid"))
print("RFD3 rows:", json.dumps(rfd3, indent=2, sort_keys=True))

ok = bool(rfd3) and all(v.get("resolved") == "not-imported" for v in rfd3.values())
base = {f: v for f, v in levers.items() if not f.startswith("RFD3_")}
findings = rg._size_ladder_compare_levers(base, levers, "256aa")
print("levers censused:", len(levers), "| RFD3 rows:", sorted(rfd3))
print("findings when the baseline predates the two RFD3 rows:", findings)
print("VERDICT:", "PASS" if (ok and findings == []) else "FAIL")
json.dump({"rfd3_rows": rfd3, "all_not_imported": ok, "findings": findings,
           "runtime_s": r.get("runtime_s"), "card": "2 (p300c, tt-quietbox2)"},
          open(REPO / "perf" / "p137" / "boltz2_rfd3_rows.json", "w"), indent=2, sort_keys=True)
sys.exit(0 if (ok and findings == []) else 1)
