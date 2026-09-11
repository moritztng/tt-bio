"""Record protenix-v2's ladder into the real baseline, with SIZE_LADDER_WORKDIR pointed at a
private directory.

Attempt 2 of chain5. Attempt 1 is the plain CLI, which is the configuration that has now
crashed twice: both times protenix-v2's ladder ran directly after another model's ladder in the
shared perf/sizegate/work, and both times the whole workdir vanished mid-fold (watched at
1 Hz on 2026-09-10: present 10:08:44, gone 10:08:45, 17 s into rep0 at rung 256). Walking the
same six rungs in a private workdir did not reproduce it. This is the recorder's own code path,
only the scratch directory differs.
"""
import importlib.util, json, sys, time
from pathlib import Path

REPO = Path("/home/ttuser/.coworker/wt/capacity-size-ladder-reseed-p300c")
spec = importlib.util.spec_from_file_location("rg", REPO / "scripts" / "release_gate.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)

rg.SIZE_LADDER_WORKDIR = REPO / "perf" / "capacity-size-ladder-reseed-p300c" / "pv2work"
t0 = time.time()
row = rg.run_size_ladder(False, True, REPO / "docs" / "size_ladder_baseline.json",
                         models=["protenix-v2"])
print(f"\nrun_size_ladder returned after {time.time()-t0:.1f}s")
print(json.dumps({k: v for k, v in row.items() if k != "legs"}, indent=2, default=str))
for leg in row.get("legs") or []:
    print("leg:", {k: v for k, v in leg.items() if k in ("model", "gate", "error")})
sys.exit(0 if not row.get("error") else 1)
