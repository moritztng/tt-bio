"""Aggregate DockQ across all folded 2026ARK-AB pilot targets (run on qb2, worktree PYTHONPATH)."""
import json, subprocess, sys, os

TARGETS = ["9ck4", "9d3j", "9i5n", "9m72", "9obn", "22ps", "9yio", "9ncy", "9w14", "9gfr", "9udq", "9jkr"]
OUT_BASE = "/tmp/abag_pilot_out"
GT = "examples/ground_truth_structures"

summary = {}
for t in TARGETS:
    cif = f"{OUT_BASE}/{t}_s1/opendde_results_{t}_abag/structures/{t}_abag.cif"
    native = f"{GT}/{t}.cif"
    if not os.path.exists(cif):
        summary[t] = {"status": "not folded"}
        continue
    r = subprocess.run([sys.executable, "scripts/opendde_dockq.py", cif, native],
                        capture_output=True, text=True)
    if r.returncode not in (0, 2):
        summary[t] = {"status": f"dockq error rc={r.returncode}", "stderr": r.stderr[-500:]}
        continue
    try:
        d = json.loads(r.stdout)
    except Exception as e:
        summary[t] = {"status": f"parse error: {e}", "stdout": r.stdout[-500:]}
        continue
    # antigen-antibody interfaces = all EXCEPT the H-L (internal Fab) interface;
    # heuristically: the interface with the smaller of the two chain lengths being
    # the antigen is ambiguous here, so just report ALL interfaces + flag the max DockQ
    # among interfaces NOT matching both antibody chains (H&L).
    ifaces = d.get("interfaces", {})
    summary[t] = {"status": "scored", "global_dockq": d["global_dockq"],
                  "n_interfaces": d["n_interfaces"], "interfaces": ifaces,
                  "chain_map": d["chain_map"]}

json.dump(summary, open("/tmp/abag_pilot_out/stage1_summary.json", "w"), indent=2)
for t, s in summary.items():
    print(t, s.get("status"), s.get("global_dockq"), s.get("n_interfaces"))
