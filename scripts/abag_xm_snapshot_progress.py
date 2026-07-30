#!/usr/bin/env python3
"""Snapshot each host's completed-fold list to the orchestrating side, so losing a host does not lose
the ability to resume around it.

Written the hard way. qb2 went hard-down on 2026-07-28 holding ~125 completed folds. The data was on
its disk -- not lost, just unreachable -- but its `progress.jsonl` was unreachable too, so there was no
way to know WHICH pairs it had finished. That turned the cheap recovery (have the surviving host fold
only the missing pairs) into a choice between re-folding ~125 completed folds or waiting indefinitely.
Counts had been recorded in the state doc all along; the pair list never was.

Writes one JSON per host per run, plus a `latest.json` symlink-equivalent, under
~/.coworker/state/abag_xm_snapshots/. An unreachable host is recorded as unreachable and its previous
snapshot is left alone, so the last known good list survives an outage.

    abag_xm_snapshot_progress.py [host ...]      # default: tt-quietbox tt-quietbox2
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HOSTS = sys.argv[1:] or ["tt-quietbox", "tt-quietbox2"]
OUT = Path.home() / ".coworker" / "state" / "abag_xm_snapshots"
OUT.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M%S")

REMOTE = (
    "python3 -c \""
    "import json;"
    "rs=[json.loads(l) for l in open('/home/ttuser/abag_xm/tier_a/progress.jsonl') if l.strip()];"
    "ok=sorted({(r['target'],r['model']) for r in rs if r.get('status')=='ok'});"
    "print(json.dumps({'n_records':len(rs),'ok':ok}))\""
)

summary = {}
for h in HOSTS:
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        f"ttuser@{h}", REMOTE], capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not r.stdout.strip():
        prev = sorted(OUT.glob(f"{h}-*.json"))
        note = f"unreachable; last snapshot {prev[-1].name}" if prev else "unreachable; NO prior snapshot"
        print(f"{h}: {note}")
        summary[h] = {"reachable": False, "note": note}
        continue
    d = json.loads(r.stdout)
    ok_pairs = [list(p) for p in d["ok"]]
    path = OUT / f"{h}-{STAMP}.json"
    path.write_text(json.dumps({"host": h, "stamp": STAMP, "n_records": d["n_records"],
                                "n_ok": len(ok_pairs), "ok": ok_pairs}, indent=1))
    (OUT / f"{h}-latest.json").write_text(path.read_text())
    print(f"{h}: {len(ok_pairs)} ok pairs, {d['n_records']} records -> {path.name}")
    summary[h] = {"reachable": True, "n_ok": len(ok_pairs), "n_records": d["n_records"]}

# Union across whatever is reachable plus the last known list for whatever is not, which is the number
# that actually matters and the thing an outage otherwise destroys.
union = set()
for h in HOSTS:
    p = OUT / f"{h}-latest.json"
    if p.exists():
        union |= {tuple(x) for x in json.loads(p.read_text())["ok"]}
print(f"\nunion over latest snapshots (reachable or not): {len(union)} / 492 pairs ok")
unreachable = [h for h, v in summary.items() if not v["reachable"]]
if unreachable:
    print(f"using last known list for: {', '.join(unreachable)} -- so a takeover can fold only the "
          f"missing pairs instead of everything")

# Retain a bounded history: newest 20 per host.
for h in HOSTS:
    snaps = sorted(OUT.glob(f"{h}-2*.json"))
    for old in snaps[:-20]:
        old.unlink()
