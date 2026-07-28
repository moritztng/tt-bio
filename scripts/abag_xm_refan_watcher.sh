#!/usr/bin/env bash
# Autonomous re-fan watcher for the abag-xm Tier-A campaign.
# Loops every 60s; when a card (0-3) has NO active abag_xm_generate.py, launches a
# fresh slice of up to 20 remaining targets for the most-lagging generator.
# Logs to ~/abag_xm/tier_a/logs/refan_watcher.log
# Idempotent: skips targets already "ok" in progress.jsonl (the harness also skips).
set -u
WT=/home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p3
TIER=/home/ttuser/abag_xm/tier_a
LOG=$TIER/logs/refan_watcher.log
LEASE=worker:abag-xm-crossmodel-ranking-dataset-p3
TARGETS_PARQUET=$WT/docs/implementation-parity-data/abag-xm-targets.parquet

mkdir -p "$TIER/logs"

# full target list (164) — system python3 has pyarrow; deeprank_ab_venv does not
ALL_TGT=$(python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("/home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p3/docs/implementation-parity-data/abag-xm-targets.parquet")
print(",".join(df.iloc[:,0].tolist()))
PY
)

remaining() {  # $1 = model name (boltz2|opendde-abag|protenix-v2); echoes comma-sep remaining targets
  # Excludes targets already "ok" AND targets currently in-progress (in any active
  # harness's --targets list for the SAME model) to avoid duplicate folds.
  python3 - "$1" "$ALL_TGT" <<'PY'
import json, os, re, subprocess, sys
model, allcsv = sys.argv[1], sys.argv[2]
allt = allcsv.split(",")
ok = set()
try:
    with open("/home/ttuser/abag_xm/tier_a/progress.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["model"] == model and r["status"] == "ok":
                ok.add(r["target"])
except FileNotFoundError:
    pass
# in-progress: targets in any active abag_xm_generate.py --targets list for this model
inprog = set()
try:
    out = subprocess.check_output(["pgrep", "-af", "abag_xm_generate.py"], text=True)
except subprocess.CalledProcessError:
    out = ""
for line in out.splitlines():
    if "pgrep" in line or "--targets" not in line:
        continue
    # match --models <model> ... --targets <csv>
    m = re.search(r"--models\s+(\S+)", line)
    t = re.search(r"--targets\s+([^\s]+)", line)
    if m and t and m.group(1) == model:
        for tgt in t.group(1).split(","):
            inprog.add(tgt)
rem = [t for t in allt if t not in ok and t not in inprog]
print(",".join(rem))
PY
}

card_busy() {  # $1 = device id; returns 0 if busy, 1 if free
  pgrep -af "abag_xm_generate.py" | grep -v pgrep | grep -q -- "--device $1 "
}

launch_slice() {  # $1 = device, $2 = model, $3 = targets csv
  local dev=$1 mdl=$2 tgts=$3
  # take first 20 targets
  local slice=$(echo "$tgts" | tr ',' '\n' | head -20 | tr '\n' ',' | sed 's/,$//')
  [ -z "$slice" ] && return 1
  local name="refan_${mdl}_${dev}_$(date +%H%M)"
  echo "[$(date +%H:%M)] launching $mdl on card$dev: $slice" >> "$LOG"
  cd "$WT"
  TT_VISIBLE_DEVICES=$dev TT_BIO_LEASE_HOLDER=$LEASE \
    nohup python3 scripts/abag_xm_generate.py --device $dev --models $mdl \
      --targets "$slice" --timeout 7200 \
      > "$TIER/logs/${name}.log" 2>&1 &
  disown -a
  echo "[$(date +%H:%M)] launched pid $!" >> "$LOG"
  return 0
}

echo "[$(date +%H:%M)] refan watcher started (ALL_TGT=$ALL_TGT)" >> "$LOG"

while true; do
  # priority: most-lagging generator first. Count remaining per gen.
  B_REM=$(remaining boltz2)
  O_REM=$(remaining opendde-abag)
  P_REM=$(remaining protenix-v2)
  B_N=$(echo "$B_REM" | tr ',' '\n' | grep -c . || true)
  O_N=$(echo "$O_REM" | tr ',' '\n' | grep -c . || true)
  P_N=$(echo "$P_REM" | tr ',' '\n' | grep -c . || true)

  # try each free card, pick the gen with most remaining (tie -> protenix > opendde > boltz2)
  for dev in 0 1 2 3; do
    if ! card_busy "$dev"; then
      # choose gen
      gen=""
      tgts=""
      if [ "$P_N" -ge "$O_N" ] && [ "$P_N" -ge "$B_N" ] && [ -n "$P_REM" ]; then
        gen=protenix-v2; tgts=$P_REM
      elif [ "$O_N" -ge "$B_N" ] && [ -n "$O_REM" ]; then
        gen=opendde-abag; tgts=$O_REM
      elif [ -n "$B_REM" ]; then
        gen=boltz2; tgts=$B_REM
      fi
      if [ -n "$gen" ]; then
        launch_slice "$dev" "$gen" "$tgts" && sleep 5
      fi
    fi
  done

  # all done?
  if [ "$B_N" -eq 0 ] && [ "$O_N" -eq 0 ] && [ "$P_N" -eq 0 ]; then
    echo "[$(date +%H:%M)] ALL 492 FOLDS COMPLETE; watcher exiting" >> "$LOG"
    break
  fi
  sleep 60
done
