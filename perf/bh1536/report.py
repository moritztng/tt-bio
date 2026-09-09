#!/usr/bin/env python3
"""results.jsonl -> the markdown table that goes in FINDINGS.md, and the per-model verdict.

Kept as code rather than a hand-typed table because the ladder runs across several relaunches
and a hand-copied number drifts from the log it came from.
"""
import fcntl
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
JL = HERE / "results.jsonl"
rows = [json.loads(l) for l in JL.read_text().splitlines() if l.strip()]

if "--backfill" in sys.argv:
    lock = (HERE / ".results.lock").open("w")   # separate file: the jsonl is opened "a" by the
    fcntl.flock(lock, fcntl.LOCK_EX)            # writer, and scoring can take minutes
    # Rungs recorded before run_rung.py started scoring geometry still have their output dirs.
    # Score them from the same CIF rather than leaving the column empty or, worse, guessing.
    changed = False
    for r in rows:
        if r.get("struct") or r["verdict"] != "PASS" or not r.get("cif"):
            continue
        run_dir = Path(r["cif"])
        while run_dir.parent.name != "runs" and run_dir.parent != run_dir:
            run_dir = run_dir.parent
        out = subprocess.run([PY, str(WT / "perf/ceilings/struct_signal.py"), str(run_dir)],
                             capture_output=True, text=True, cwd=str(WT),
                             env={"PYTHONPATH": str(WT), "PATH": "/usr/bin:/bin"})
        try:
            r["struct"] = json.loads(out.stdout.strip().splitlines()[-1])
            changed = True
            print(f"scored {r['model']} {r['size']}: {r['struct']}", file=sys.stderr)
        except Exception as exc:
            print(f"could not score {r['model']} {r['size']}: {exc} {out.stderr[-200:]}",
                  file=sys.stderr)
    # Re-read under the lock: a rung may have landed while struct_signal was scoring, and
    # rewriting from the pre-scan snapshot would delete it.
    fresh = [json.loads(l) for l in JL.read_text().splitlines() if l.strip()]
    scored = {(r["model"], r["size"], r.get("tag", ""), r["when"]): r.get("struct")
              for r in rows if r.get("struct")}
    for r in fresh:
        key = (r["model"], r["size"], r.get("tag", ""), r["when"])
        if not r.get("struct") and scored.get(key):
            r["struct"] = scored[key]; changed = True
    if changed:
        JL.write_text("".join(json.dumps(r) + "\n" for r in fresh))
    rows = fresh
    fcntl.flock(lock, fcntl.LOCK_UN)

rows = [r for r in rows if r.get("tag") != "harnesscheck"]

by_model = {}
for r in rows:
    by_model.setdefault(r["model"], []).append(r)

print("| model | tokens | verdict | wall s | engine s | atoms | CA breaks | worst CA-CA | clash frac | pLDDT | host RSS |")
print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for m in sorted(by_model):
    for r in sorted(by_model[m], key=lambda x: x["size"]):
        s = r.get("struct") or {}
        print(f"| {m} | {r['size']} | {r['verdict']} | {r['wall_s']} | "
              f"{r.get('engine_runtime_s') or s.get('runtime_s') or '-'} | "
              f"{s.get('n_atoms', '-')} | {s.get('ca_breaks', '-')} | "
              f"{s.get('worst_ca_ca', '-')} | {s.get('clash_frac', '-')} | "
              f"{r.get('plddt') or s.get('plddt') or r.get('affinity') or '-'} | "
              f"{r['peak_host_rss_gib']} |")

print()
for m in sorted(by_model):
    rs = sorted(by_model[m], key=lambda x: x["size"])
    top = max(rs, key=lambda x: (x["verdict"] == "PASS", x["size"]))
    passes = [r for r in rs if r["verdict"] == "PASS"]
    fails = [r for r in rs if r["verdict"] != "PASS"]
    hi = max((r["size"] for r in passes), default=None)
    lo = min((r["size"] for r in fails), default=None)
    print(f"{m}: pass<={hi} fail>={lo} rungs={[ (r['size'], r['verdict']) for r in rs ]}")
