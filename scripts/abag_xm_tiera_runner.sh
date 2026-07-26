#!/usr/bin/env bash
# abag_xm_tiera_runner.sh — Tier-A driver for qb1 (4 cards). Conservative: runs
# generate.py per card with --timeout 3600 (generate.py's own per-fold cap handles
# any single hang). A 30-min no-progress safety net (across ALL cards) catches a
# genuine dirty-chip deadlock and resets all boards + relaunches. 50-sample folds
# take ~3-10 min, so 30 min of zero progress.jsonl updates = real hang, NOT a slow
# fold. generate.py skips ok pairs (progress.jsonl status==ok), so recovery never
# re-does or loses an ok fold. LESSON from earlier passes: do NOT kill folds based
# on parent-process CPU (the predict parent sits in do_wait at 0 CPU while its
# multiprocessing child worker does the real device work at 4x CPU) — that killed
# slow-but-progressing folds prematurely.
set +u
WT=/home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p3
PY=/home/ttuser/tt-bio/env/bin/python3
PROG=$WT/scripts/abag_xm_generate.py
PROGRESS=$HOME/abag_xm/tier_a/progress.jsonl
LEASE=worker:abag-xm-crossmodel-ranking-dataset-p3
TIMEOUT=3600
STALL=1800          # 30 min no-progress -> genuine hang -> reset all + relaunch
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
kill_all(){ for p in "${PIDS[@]}"; do [ -n "$p" ] && kill -INT "$p" 2>/dev/null; done; sleep 6
  for c in $(pgrep -f "python3 -m tt_bio.main predict" 2>/dev/null); do kill -KILL "$c" 2>/dev/null; done
  for c in $(pgrep -f "python3.*abag_xm_generate" 2>/dev/null); do kill -KILL "$c" 2>/dev/null; done
  for c in $(pgrep -f "multiprocessing.spawn import spawn_main" 2>/dev/null); do kill -KILL "$c" 2>/dev/null; done
  sleep 2; rm -f ~/.coworker/state/leases/tt-quietbox-card*.json; }
recover_all(){ log "RECOVER (30min stall): kill all + tt-smi -r 0,1,2,3 + relaunch"
  kill_all; timeout 120 "$TT_SMI" -r 0,1,2,3 >/dev/null 2>&1; sleep 3
  PIDS=(); for c in 0 1 2 3; do PIDS[$c]=$(launch_card $c); log "relaunched card $c pid=${PIDS[$c]}"; done; }
progress_mtime(){ stat -c %Y "$PROGRESS" 2>/dev/null || echo 0; }

mkdir -p "$HOME/abag_xm/tier_a"
log "START Tier-A runner v3: 4 cards, timeout=${TIMEOUT}s, stall=${STALL}s (conservative)"
PIDS=(); for c in 0 1 2 3; do PIDS[$c]=$(launch_card $c); log "launched card $c pid=${PIDS[$c]}"; done
last_prog=$(progress_mtime)
while true; do
  sleep 120
  alive=0; for c in 0 1 2 3; do [ -n "${PIDS[$c]}" ] && kill -0 "${PIDS[$c]}" 2>/dev/null && alive=$((alive+1)); done
  if [ "$alive" -eq 0 ]; then log "all 4 generate.py exited — campaign slice complete"; break; fi
  now_prog=$(progress_mtime)
  if [ "$now_prog" != "$last_prog" ]; then last_prog=$now_prog
  elif [ $(( $(date +%s) - now_prog )) -ge "$STALL" ]; then recover_all; last_prog=$(progress_mtime); fi
done
log "DONE"
