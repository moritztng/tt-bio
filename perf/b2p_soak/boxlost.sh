#!/bin/bash
# The box-lost drill: a campaign killed outright at trajectory N, then restarted on its folder.
#
# A researcher loses a box mid-campaign -- a reset, an OOM kill, a power event -- and the claim
# production readiness has to make is that every design already accepted survives it and nothing
# is charged twice. That is a diff, not a promise, so this takes a census either side and lets
# census.py deliver the verdict as an exit code. SIGKILL to the whole process group is the honest
# stand-in: nothing gets to flush, exactly as nothing does when the box goes.
#
# Armed rather than timed. It waits for the campaign's own state file to say the target
# trajectory is charged, so the interruption lands where the drill says it does and not where a
# guessed wall clock happened to fall. Runs for hours; start it detached.
#
#   TARGET=12 boxlost.sh          # interrupt once 12 trajectories are charged
#
# Everything it needs comes from the environment launch.sh already uses, plus:
#   SOAK_LIVE     the running campaign's output dir (its pid file and project live there)
#   SOAK_DRILL    where the censuses and this log go
set -uo pipefail
: "${SOAK_VENV:?}" "${SOAK_BC2:?}" "${SOAK_PARAMS:?}" "${SOAK_REPO:?}" "${SOAK_CARD:?}"
live=${SOAK_LIVE:-$HOME/b2p_soak/out/long24}
drill=${SOAK_DRILL:-$HOME/b2p_soak/out/boxlost}
target=${TARGET:-12}
deadline=$(( $(date +%s) + ${DEADLINE_HOURS:-9} * 3600 ))
project=$live/project
census=${SOAK_CENSUS:-$(dirname "$0")/census.py}
launch=$(dirname "$0")/launch.sh
mkdir -p "$drill"
log=$drill/boxlost.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }

charged() {
  python3 - "$project/.campaign_state.json" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    print(int(json.load(open(sys.argv[1]))["trajectories"]))
except Exception:
    print(0)
PY
}
lease_file() { ls "${TT_BIO_LEASE_DIR:-$HOME/.tt_bio_leases}"/*card"$SOAK_CARD".json 2>/dev/null | head -1; }

say "armed: interrupt $live at $target charged trajectories, card $SOAK_CARD"
while :; do
  pid=$(cat "$live/pid" 2>/dev/null || echo 0)
  now=$(charged)
  if ! kill -0 "$pid" 2>/dev/null; then
    say "campaign pid $pid is already gone at $now charged; there is nothing to interrupt. \
Either it finished or it died on its own -- read $live/campaign.log before treating this as a \
drill result."
    exit 3
  fi
  [ "$now" -ge "$target" ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "deadline reached at $now of $target charged; NOT interrupting. A drill that fires late \
proves nothing about trajectory $target."
    exit 4
  fi
  sleep 60
done

say "trajectory $now charged; census before the interruption"
"$SOAK_VENV" "$census" "$project" --out "$drill/before.json" >> "$log" 2>&1
cp -a "$project/.campaign_state.json" "$drill/state-before.json" 2>/dev/null
leased=$(lease_file); say "lease before: ${leased:-none} $(cat "$leased" 2>/dev/null)"

say "SIGKILL to process group $pid -- the box is lost"
kill -9 -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
kill -0 "$pid" 2>/dev/null && say "WARNING pid $pid survived SIGKILL" || say "campaign is dead"
say "lease after the kill: $(cat "$(lease_file)" 2>/dev/null || echo none)"

# Nothing ran between the two, so anything that moved here was half-written when the box went.
say "census immediately after the kill (nothing should have changed)"
"$SOAK_VENV" "$census" "$project" --out "$drill/after-kill.json" --against "$drill/before.json" \
  >> "$log" 2>&1
say "after-kill verdict: exit $?"

say "restarting on the same folder with resume=true"
SOAK_PROJECT=$project SOAK_OUT=${SOAK_OUT:-$HOME/b2p_soak/out} \
  bash "$launch" "${RESUME_TAG:-long24r}" --trajectories 2 --max-trajectories "${BUDGET:-24}" \
  --binder "${BINDER:-146}" --seed "${SEED:-100}" \
  --set number_of_final_designs="${FINAL:-500}" --set resume=true >> "$log" 2>&1
resumed=${SOAK_OUT:-$HOME/b2p_soak/out}/${RESUME_TAG:-long24r}
say "resumed leg logs $resumed/campaign.log pid $(cat "$resumed/pid" 2>/dev/null)"

# The resumption line is the first thing that shows what it inherited; then wait for it to charge
# past where the interruption left it, which is the evidence it carried on rather than restarted.
for _ in $(seq 180); do grep -q "resuming " "$resumed/campaign.log" 2>/dev/null && break; sleep 10; done
say "resumption line: $(grep -m1 -A3 'resuming ' "$resumed/campaign.log" 2>/dev/null | tr '\n' ' ')"
for _ in $(seq 360); do [ "$(charged)" -gt "$now" ] && break; sleep 30; done
say "charged after the resume: $(charged) (was $now)"
"$SOAK_VENV" "$census" "$project" --out "$drill/after-resume.json" --against "$drill/before.json" \
  >> "$log" 2>&1
say "after-resume verdict: exit $? -- the last RESUME line in this log is the answer"
