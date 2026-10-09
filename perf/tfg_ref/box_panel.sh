#!/usr/bin/env bash
# The SAbDab panel on the box, end to end: screens, as pc_sync_panel.sh streams targets in, all targets (5 unguided seed-101
# samples), keeps the first N_FAIL targets in panel order whose median DockQ < 0.23, then runs seeds 101-105 x
# {unconstrained, contact, pocket} on them (screening stops early once N_FAIL are found), records fixtures for two of them and scores everything.
set -euo pipefail
N_FAIL=${N_FAIL:-40}
until [ -f /root/panel/panel.json ] && grep -q EXAMPLES_DONE /root/examples.log 2>/dev/null; do sleep 30; done
cd /root
python3 -c "import json; print('\n'.join(json.load(open('/root/panel/panel.json'))['targets']))" > all.txt
# Screen in chunks of 20 in panel order and stop once N_FAIL failing targets are found (~34% fail, so ~120 screened).
split -l 20 -d all.txt chunk_
for c in chunk_*; do
  # The pc pushes the panel as MSAs finish; a chunk starts once each of its inputs points at MSA files that arrived.
  until python3 - "$c" <<'PY'
import json, os, sys
for t in open(sys.argv[1]).read().split():
    job = json.load(open(f"/root/panel/{t}/{t}_unconstrained.json"))[0]
    paths = [v for s in job["sequences"] for k, v in s["proteinChain"].items() if k.endswith("MsaPath")]
    if not paths or not all(os.path.exists(f"/root/panel/{p}") for p in paths):
        sys.exit(1)
PY
  do [ -f /root/panel/READY ] && { echo "chunk $c lacks MSAs after the final sync"; break; }; sleep 60; done
  bash box_run.sh /root/panel /root/runs/panel/screen_$c $c 101 unconstrained
  /root/dq/bin/python score.py /root/panel /root/runs/panel /root/scores.jsonl
  /root/dq/bin/python - "$N_FAIL" <<'PY'
import json, sys, statistics
rows = [json.loads(l) for l in open("/root/scores.jsonl")]
order = open("/root/all.txt").read().split()
med = {t: statistics.median([r["dockq"] for r in rows if r["target"] == t and r["cond"] == "unconstrained" and r["seed"] == 101])
       for t in order if any(r["target"] == t and r["seed"] == 101 for r in rows)}
fail = [t for t in order if t in med and med[t] < 0.23]
json.dump({"median_dockq_seed101": med, "failing": fail, "screened": len(med)}, open("/root/screen.json", "w"), indent=1)
open("/root/fail.txt", "w").write("\n".join(fail[: int(sys.argv[1])]) + "\n")
print(f"screened {len(med)}, failing {len(fail)} ({len(fail)/max(1,len(med)):.0%}), kept {min(len(fail), int(sys.argv[1]))}", flush=True)
PY
  [ "$(grep -c . fail.txt)" -ge "$N_FAIL" ] && break
done
# Seed-101 unguided candidates come from the screen.
bash box_run.sh /root/panel /root/runs/panel/full fail.txt "102 103 104 105" unconstrained
bash box_run.sh /root/panel /root/runs/panel/full fail.txt "101 102 103 104 105" "contact pocket"
for t in $(head -2 fail.txt); do
  echo "$t" > fx.txt
  for cond in contact pocket; do
    FIXTURE_DIR=/root/fixtures FIXTURE_NAME_PREFIX=${t}_${cond}_ OPENDDE_CMD="python3 /root/capture.py" \
      bash box_run.sh /root/panel /root/runs/fixtures_$t fx.txt 101 "$cond"
  done
done
/root/dq/bin/python score.py /root/panel /root/runs/panel /root/scores.jsonl
/root/dq/bin/python summarize.py /root/scores.jsonl fail.txt > /root/summary.json
echo PANEL_DONE
