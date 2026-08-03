#!/usr/bin/env bash
# p25_harvest.sh -- pull successful galaxy deep-N folds into the qb1 analysis arm tree.
#
# Runs ON qb1. Reads the run's fleet results.jsonl over ssh, rsyncs every (model,target)
# with an ok record at the requested rung(s) into ~/abag_xm/deepn/galaxy/<mdir>/<t>_n<N>/,
# verifies the harvested fold is complete on disk (results.json ok + one CIF per all_runs
# entry), and drops incomplete harvests (e.g. all-attempts-OOM targets whose on-disk state
# is the last failed attempt). Also refreshes galaxy/fleet_results.jsonl (line-deduped).
#
# usage: p25_harvest.sh <run_id> [rung ...]     e.g.: p25_harvest.sh p25 64
# On qb1 run it directly; qb1 cannot reach the galaxy (no cloudflared), so the relay is:
#   pc:  bash scripts/abag_xm/p25_harvest.sh p25 64 && bash scripts/abag_xm/p25_harvest.sh p25b 64
#   pc:  rsync -az ~/abag_xm/deepn/galaxy/ qb1:abag_xm/deepn/galaxy/
set -uo pipefail
RUN=${1:?run id, e.g. p25 or p25b}
shift
RUNGS=${*:-64}
DEST=${DEST:-$HOME/abag_xm/deepn/galaxy}
GAL=${GAL:-japanfold-ssh}
GB=/home/cust-team/mthuening/$RUN
mkdir -p "$DEST"

timeout 120 ssh -o BatchMode=yes "$GAL" "cat $GB/results.jsonl" > "$DEST/.fleet.$RUN.jsonl" \
  || { echo "WARN: no results.jsonl for $RUN"; exit 0; }
{ [ -f "$DEST/fleet_results.jsonl" ] && cat "$DEST/fleet_results.jsonl"; cat "$DEST/.fleet.$RUN.jsonl"; } \
  | sort -u > "$DEST/.fleet.all" && mv "$DEST/.fleet.all" "$DEST/fleet_results.jsonl"
rm -f "$DEST/.fleet.$RUN.jsonl"

python3 - "$DEST" "$RUN" $RUNGS <<'PY'
import json, pathlib, subprocess, sys
dest = pathlib.Path(sys.argv[1]); run = sys.argv[2]
rungs = {int(x) for x in sys.argv[3:]}
MD = {"boltz2": "boltz2", "opendde-abag": "opendde",
      "protenix-v2": "protenix", "esmfold2": "esmfold2"}
ok = {}
for line in (dest / "fleet_results.jsonl").read_text().splitlines():
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    if r.get("rc") == 0 and r.get("cifs", 0) > 0 and r.get("rung") in rungs:
        ok[(r["model"], r["target"], r["rung"])] = True
print(f"harvest: {len(ok)} ok folds at rungs {sorted(rungs)}")
gb = f"/home/cust-team/mthuening/{run}"
for (model, t, rung) in sorted(ok):
    mdir = MD[model]
    out = dest / mdir / f"{t}_n{rung}"
    rd = out / f"{mdir}_results_{t}"
    rj = rd / "results.json"
    if rj.exists():
        try:
            rec = json.loads(rj.read_text())[0]
            n = len(rec.get("all_runs") or [])
            if rec.get("status") == "ok" and n > 0 \
               and len(list((rd / "structures").glob("*.cif"))) == n:
                continue  # already harvested and complete
        except Exception:
            pass
    out.mkdir(parents=True, exist_ok=True)
    src = f"japanfold-ssh:{gb}/{mdir}/{t}/{mdir}_results_{t}/"
    r = subprocess.run(["rsync", "-az", "--timeout=300", src, str(rd) + "/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {model} {t} n{rung}: rsync FAILED {r.stderr.strip()[:120]}")
        continue
    try:
        rec = json.loads(rj.read_text())[0]
        n = len(rec.get("all_runs") or [])
        n_cif = len(list((rd / "structures").glob("*.cif")))
        if rec.get("status") == "ok" and n > 0 and n_cif == n:
            print(f"  {model} {t} n{rung}: ok ({n} cifs)")
        else:
            print(f"  {model} {t} n{rung}: INCOMPLETE on galaxy "
                  f"(status={rec.get('status')} runs={n} cifs={n_cif}) -- dropped")
            subprocess.run(["rm", "-rf", str(out)])
    except Exception as e:
        print(f"  {model} {t} n{rung}: verify error {e} -- dropped")
        subprocess.run(["rm", "-rf", str(out)])
PY
