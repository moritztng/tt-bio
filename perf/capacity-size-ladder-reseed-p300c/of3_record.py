"""Record openfold3's p300c size ladder: the last (card, model) row missing 896/1024 anywhere.

Private SIZE_LADDER_WORKDIR, which is the configuration that has walked clean every time an
undiagnosed deleter took the shared perf/sizegate/work mid-fold (openbind died at rung 640 that
way on 2026-09-10). fragment=True so the row lands in docs/size_ladder_baseline.d/openfold3.json,
the shape main converged on, instead of a monolith row a fragment would later shadow.
"""
import importlib.util, json, time
from pathlib import Path

REPO = Path("/home/ttuser/.coworker/wt/capacity-size-ladder-reseed-p300c")
spec = importlib.util.spec_from_file_location("rg", REPO / "scripts" / "release_gate.py")
rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)
rg.SIZE_LADDER_WORKDIR = REPO / "perf" / "capacity-size-ladder-reseed-p300c" / "work-openfold3"

t0 = time.time()
row = rg.run_size_ladder(False, True, REPO / "docs" / "size_ladder_baseline.json",
                         models=["openfold3"], fragment=True)
print(f"\nrun_size_ladder returned after {time.time() - t0:.1f}s", flush=True)
print(json.dumps({k: v for k, v in row.items() if k != "legs"}, indent=2, default=str))
for leg in row.get("legs") or []:
    print("leg:", {k: v for k, v in leg.items() if k in ("model", "gate", "error")})
