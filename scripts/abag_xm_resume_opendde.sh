#!/usr/bin/env bash
# abag_xm_resume_opendde.sh — resume Tier-A generation with opendde-abag after
# the protenix-v2+boltz2 fanout completes. Run this AFTER the current watchdog
# exits (protenix+boltz2 done). generate.py's done_pairs() skips the ~87 ok
# protenix+boltz2 pairs, so this only runs opendde for all 164 + retries the
# 13 timed_out + failed protenix/boltz2 pairs.
#
# Timeout raised to 7200s (was 3600) so the 13 large protenix targets that
# timed out at 3600s get another hour. Stall raised to 9000s (2.5h) so the
# watchdog doesn't kill legitimate long folds. If the 13 still time out at
# 7200s, accept the 8% protenix gap (boltz2+opendde cover those targets).
set +u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this checkout, not a hardcoded slug
PY=/home/ttuser/tt-bio/env/bin/python3
PROG=$WT/scripts/abag_xm_generate.py
PROGRESS=$HOME/abag_xm/tier_a/progress.jsonl
LEASE="worker:$(basename "$WT")"
TIMEOUT=7200
STALL=9000
TT_SMI=$HOME/.tenstorrent-venv/bin/tt-smi
# Cards this supervisor OWNS. Everything destructive below is scoped to these, because a
# QuietBox is shared with the rest of the fleet: `tt-smi -r $CARDS_CSV` on a stall would hard-reset
# a sibling worker's in-flight card, and `rm -f .../leases/tt-quietbox-card*.json` would delete
# its lease. Observed live 2026-07-27: worker tt-bio-rfdiffusion3-batch-perf-p17 held card 1
# while this campaign was idle. Override with CARDS="0 2" for a partial box.
CARDS="${CARDS:-0 1 2 3}"
CARDS_CSV=$(echo "$CARDS" | tr " " ",")
# Drop only OUR lease files, and only if we are still the recorded holder -- never a sibling's.
release_own_leases(){ local c
  for c in $CARDS; do
    local f=$HOME/.coworker/state/leases/$(hostname)-card$c.json
    [ -f "$f" ] || continue
    grep -q "\"holder\": \"$LEASE\"" "$f" && rm -f "$f"
  done; }


T0="21av,9ck4,9d74,9gfr,9i5n,9j87,9kwy,9l9y,9lh2,9log,9lr1,9lxp,9ly6,9m0j,9m2o,9m40,9ma0,9mnt,9mz6,9mzf,9n1p,9n8i,9nkz,9nw4,9nzf,9pso,9q6y,9qqf,9rn6,9sbb,9th6,9u5r,9ugo,9ulp,9v0x,9vmo,9wb3,9x05,9xqc,9y0a,9yxd"
T1="21du,9d3j,9dsg,9gvn,9iar,9jkr,9l1l,9lbw,9lme,9loz,9lsy,9ly2,9lz0,9m0x,9m2s,9m72,9mmj,9mnu,9mz7,9n05,9n1q,9n8n,9nl0,9nw7,9obn,9q1l,9q6z,9qrv,9rye,9ssm,9tmp,9ua5,9uk2,9uo0,9v1h,9vnp,9wb4,9x0j,9xqn,9y0e,9ynx"
T2="21tw,9d72,9fte,9hv9,9ivj,9jno,9l8z,9ldx,9loe,9lp1,9lsz,9ly3,9lz1,9m0z,9m3q,9m8k,9mnb,9msc,9mz8,9n09,9n2i,9ncd,9nl1,9nw8,9ppw,9q6h,9q7y,9rig,9ryf,9t3r,9u5p,9udq,9ull,9uoc,9ve0,9vo2,9wpm,9x3z,9xsx,9yc5,9zdu"
T3="22ps,9d73,9gei,9i3p,9j4c,9k6j,9l9o,9le0,9lof,9lqw,9lwc,9ly5,9lz2,9m1p,9m3s,9m8l,9mns,9mxu,9mze,9n0e,9n38,9ncy,9np0,9nw9,9ppy,9q6n,9qqe,9rih,9sat,9t3s,9u5q,9ue0,9ulm,9uoi,9vmn,9w14,9wwh,9xqb,9xth,9yio,9zen"
SUBSETS=("$T0" "$T1" "$T2" "$T3")

log(){ echo "[$(date +%H:%M:%S)] $*"; }
launch_card(){ local card="$1"; local targets="${SUBSETS[$card]}"
  TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_HOLDER=$LEASE PYTHONPATH=$WT \
    PYTHONUNBUFFERED=1 nohup "$PY" -u "$PROG" --targets "$targets" --device "$card" \
    --timeout "$TIMEOUT" --models protenix-v2,boltz2,opendde-abag \
    >> /tmp/abag_tiera_card$card.log 2>&1 < /dev/null &
  echo "$!"; }
kill_all(){ for c in $(pgrep -f "$WT.*tt_bio.main predict" 2>/dev/null); do kill -INT "$c" 2>/dev/null; done; sleep 6
  for c in $(pgrep -f "$WT.*tt_bio.main predict" 2>/dev/null); do kill -KILL "$c" 2>/dev/null; done
  for c in $(pgrep -f "$WT.*abag_xm_generate" 2>/dev/null); do kill -KILL "$c" 2>/dev/null; done
  for c in $(pgrep -f "multiprocessing.spawn import spawn_main" 2>/dev/null); do kill -KILL "$c" 2>/dev/null; done
  sleep 2; release_own_leases; }
recover_all(){ log "RECOVER (stall): kill all + tt-smi -r $CARDS_CSV + relaunch"
  kill_all; timeout 120 "$TT_SMI" -r "$CARDS_CSV" >/dev/null 2>&1; sleep 3
  PIDS=(); for c in 0 1 2 3; do PIDS[$c]=$(launch_card $c); log "relaunched card $c pid=${PIDS[$c]}"; done; }
progress_mtime(){ stat -c %Y "$PROGRESS" 2>/dev/null || echo 0; }

# sanity: don't start if a fanout is already running
nrun=$(pgrep -f "$WT.*abag_xm_generate" 2>/dev/null | wc -l)
if [ "$nrun" -gt 0 ]; then
  log "ABORT: $nrun generate.py already running — current fanout not done yet. Wait for the watchdog to exit."
  exit 1
fi

mkdir -p "$HOME/abag_xm/tier_a"
log "START opendde resume: 4 cards, models=protenix-v2,boltz2,opendde-abag, timeout=${TIMEOUT}s, stall=${STALL}s"
log "  (done_pairs skips the ~87 ok protenix+boltz2; runs opendde for all 164 + retries)"
PIDS=(); for c in 0 1 2 3; do PIDS[$c]=$(launch_card $c); log "launched card $c pid=${PIDS[$c]}"; done
# Stall clock starts at launch wall-clock, NOT file mtime: progress.jsonl already exists here
# (qb1's protenix+boltz2 fanout wrote it), but if it is ever cleared the mtime=0 path would
# make `now - 0` >= STALL and false-recover every tick. Wall-clock launch is robust to both.
last_prog=$(date +%s)
while true; do
  sleep 120
  alive=0; for c in 0 1 2 3; do [ -n "${PIDS[$c]}" ] && kill -0 "${PIDS[$c]}" 2>/dev/null && alive=$((alive+1)); done
  if [ "$alive" -eq 0 ]; then log "all 4 generate.py exited — campaign complete"; break; fi
  pm=$(progress_mtime)
  if [ "$pm" != "0" ] && [ "$pm" -gt "$last_prog" ]; then last_prog=$pm; fi
  now=$(date +%s)
  if [ $(( now - last_prog )) -ge "$STALL" ]; then recover_all; last_prog=$(date +%s); fi
done
log "DONE"
