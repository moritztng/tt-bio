#!/usr/bin/env bash
# abag_xm_tiera_watchdog.sh — conservative watchdog for the autonomous Tier-A
# generate.py campaign on qb1. ADOPTS already-running generate.py (does not
# relaunch/lease-collide). Only acts on a genuine 90-min no-progress stall
# (max ok fold = 54 min for protenix-v2 / 51 min for opendde, so 90 min of zero
# progress.jsonl updates = real dirty-chip deadlock, NOT a slow fold). On stall:
# kill all generate.py+predict+mp-trackers, tt-smi -r 0,1,2,3, clear leases,
# relaunch 4 fresh generate.py (skipping ok pairs). generate.py is otherwise
# autonomous (loops its 41 targets x 3 models with --timeout 3600 per fold).
set +u
WT=/home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p3
PY=/home/ttuser/tt-bio/env/bin/python3
PROG=$WT/scripts/abag_xm_generate.py
PROGRESS=$HOME/abag_xm/tier_a/progress.jsonl
LEASE=worker:abag-xm-crossmodel-ranking-dataset-p3
TIMEOUT=3600
STALL=5400          # 90 min — > max fold (54 min) so slow folds are never killed
TT_SMI=$HOME/.tenstorrent-venv/bin/tt-smi
T0="21av,9ck4,9d74,9gfr,9i5n,9j87,9kwy,9l9y,9lh2,9log,9lr1,9lxp,9ly6,9m0j,9m2o,9m40,9ma0,9mnt,9mz6,9mzf,9n1p,9n8i,9nkz,9nw4,9nzf,9pso,9q6y,9qqf,9rn6,9sbb,9th6,9u5r,9ugo,9ulp,9v0x,9vmo,9wb3,9x05,9xqc,9y0a,9yxd"
T1="21du,9d3j,9dsg,9gvn,9iar,9jkr,9l1l,9lbw,9lme,9loz,9lsy,9ly2,9lz0,9m0x,9m2s,9m72,9mmj,9mnu,9mz7,9n05,9n1q,9n8n,9nl0,9nw7,9obn,9q1l,9q6z,9qrv,9rye,9ssm,9tmp,9ua5,9uk2,9uo0,9v1h,9vnp,9wb4,9x0j,9xqn,9y0e,9ynx"
T2="21tw,9d72,9fte,9hv9,9ivj,9jno,9l8z,9ldx,9loe,9lp1,9lsz,9ly3,9lz1,9m0z,9m3q,9m8k,9mnb,9msc,9mz8,9n09,9n2i,9ncd,9nl1,9nw8,9ppw,9q6h,9q7y,9rig,9ryf,9t3r,9u5p,9udq,9ull,9uoc,9ve0,9vo2,9wpm,9x3z,9xsx,9yc5,9zdu"
T3="22ps,9d73,9gei,9i3p,9j4c,9k6j,9l9o,9le0,9lof,9lqw,9lwc,9ly5,9lz2,9m1p,9m3s,9m8l,9mns,9mxu,9mze,9n0e,9n38,9ncy,9np0,9nw9,9ppy,9q6n,9qqe,9rih,9sat,9t3s,9u5q,9ue0,9ulm,9uoi,9vmn,9w14,9wwh,9xqb,9xth,9yio,9zen"
SUBSETS=("$T0" "$T1" "$T2" "$T3")
log(){ echo "[$(date +%H:%M:%S)] $*"; }
launch_card(){ local card="$1"; local targets="${SUBSETS[$card]}"
  TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_HOLDER=$LEASE PYTHONPATH=$WT \
    PYTHONUNBUFFERED=1 nohup "$PY" -u "$PROG" --targets "$targets" --device "$card" \
    --timeout "$TIMEOUT" >> /tmp/abag_tiera_card$card.log 2>&1 < /dev/null &
  echo "$!"; }
kill_all(){ for c in $(pgrep -f "python3 -m tt_bio.main predict" 2>/dev/null); do kill -KILL "$c" 2>/dev/null; done
  for c in $(pgrep -f "python3.*abag_xm_generate" 2>/dev/null); do kill -KILL "$c" 2>/dev/null; done
  for c in $(pgrep -f "multiprocessing.spawn import spawn_main" 2>/dev/null); do kill -KILL "$c" 2>/dev/null; done
  sleep 3; rm -f ~/.coworker/state/leases/tt-quietbox-card*.json; }
progress_mtime(){ stat -c %Y "$PROGRESS" 2>/dev/null || echo 0; }

# ADOPT existing generate.py if 4 are already running (don't lease-collide)
existing=$(ps -ef | grep "python3.*abag_xm_generate" | grep -v grep | grep -v "bash -c" | awk "{print \$2}")
nexist=$(echo "$existing" | grep -c . 2>/dev/null || echo 0)
if [ "$nexist" -ge 4 ]; then
  log "ADOPT: $nexist generate.py already running — adopting, no relaunch"
else
  log "START: only $nexist generate.py running — launching 4 fresh"
  for c in 0 1 2 3; do launch_card $c >/dev/null; done
  sleep 5
fi
# Stall clock starts at launch wall-clock, NOT file mtime. progress_mtime is 0 until the
# first fold completes, and `now - 0` is the epoch, which is >= any STALL -- so seeding from
# the mtime makes a FRESH campaign (no progress.jsonl yet) kill+reset+relaunch on the very
# first tick, forever, destroying every fold before one can finish. Same fix as
# abag_xm_resume_opendde.sh / abag_xm_opendde_qb2.sh (70961944, 42ef721a); these two scripts
# were missed there and matter now that Phase 3 starts from an empty tier_a.
last_prog=$(date +%s)
zero_streak=0
log "WATCHDOG active: stall=${STALL}s (90min), timeout=${TIMEOUT}s. generate.py runs autonomously."
while true; do
  sleep 120
  ngen=$(ps -ef | grep "python3.*abag_xm_generate" | grep -v grep | grep -v "bash -c" | wc -l)
  if [ "$ngen" -eq 0 ]; then
    zero_streak=$((zero_streak+1))
    [ "$zero_streak" -ge 2 ] && { log "all generate.py exited (2 consecutive checks) — campaign complete"; break; }
    log "ngen=0 (check $zero_streak/2) — re-checking next cycle before exiting"; continue
  fi
  zero_streak=0
  pm=$(progress_mtime)
  if [ "$pm" != "0" ] && [ "$pm" -gt "$last_prog" ]; then last_prog=$pm; fi
  if [ $(( $(date +%s) - last_prog )) -ge "$STALL" ]; then
    log "STALL: no progress for ${STALL}s — kill all + tt-smi -r 0,1,2,3 + relaunch"
    kill_all; timeout 120 "$TT_SMI" -r 0,1,2,3 >/dev/null 2>&1; sleep 3
    for c in 0 1 2 3; do launch_card $c >/dev/null; done
    last_prog=$(date +%s); log "relaunched 4 generate.py"
  fi
done
log "DONE"
