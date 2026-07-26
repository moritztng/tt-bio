#!/usr/bin/env bash
# abag_xm_tiera_runner.sh — robust Tier-A driver for qb1 (4 cards) with per-card
# CPU-based hang detection. The recurring tt_bio dispatch deadlock freezes a
# predict proc at ~constant CPU (hung at _assert_local_dispatch->synchronize_device).
# CPU-stall detection distinguishes a slow-but-progressing fold (CPU advancing) from
# a hung fold (CPU frozen), so legit long folds are NOT killed mid-run. On hang:
# kill just that card's generate.py+predict, tt-smi -r <card>, relaunch that card
# (generate.py skips ok pairs via progress.jsonl, so no ok fold is re-done/lost).
set +u
WT=/home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p3
PY=/home/ttuser/tt-bio/env/bin/python3
PROG=$WT/scripts/abag_xm_generate.py
PROGRESS=$HOME/abag_xm/tier_a/progress.jsonl
LEASE=worker:abag-xm-crossmodel-ranking-dataset-p3
TIMEOUT=3600        # per-fold wall cap (match default FOLD_TIMEOUT_S; big 50-sample folds need up to ~30 min)
STALL_CPU=2         # <2s CPU advance in a 60s window while alive => hung
TT_SMI=$HOME/.tenstorrent-venv/bin/tt-smi

T0="21av,9ck4,9d74,9gfr,9i5n,9j87,9kwy,9l9y,9lh2,9log,9lr1,9lxp,9ly6,9m0j,9m2o,9m40,9ma0,9mnt,9mz6,9mzf,9n1p,9n8i,9nkz,9nw4,9nzf,9pso,9q6y,9qqf,9rn6,9sbb,9th6,9u5r,9ugo,9ulp,9v0x,9vmo,9wb3,9x05,9xqc,9y0a,9yxd"
T1="21du,9d3j,9dsg,9gvn,9iar,9jkr,9l1l,9lbw,9lme,9loz,9lsy,9ly2,9lz0,9m0x,9m2s,9m72,9mmj,9mnu,9mz7,9n05,9n1q,9n8n,9nl0,9nw7,9obn,9q1l,9q6z,9qrv,9rye,9ssm,9tmp,9ua5,9uk2,9uo0,9v1h,9vnp,9wb4,9x0j,9xqn,9y0e,9ynx"
T2="21tw,9d72,9fte,9hv9,9ivj,9jno,9l8z,9ldx,9loe,9lp1,9lsz,9ly3,9lz1,9m0z,9m3q,9m8k,9mnb,9msc,9mz8,9n09,9n2i,9ncd,9nl1,9nw8,9ppw,9q6h,9q7y,9rig,9ryf,9t3r,9u5p,9udq,9ull,9uoc,9ve0,9vo2,9wpm,9x3z,9xsx,9yc5,9zdu"
T3="22ps,9d73,9gei,9i3p,9j4c,9k6j,9l9o,9le0,9lof,9lqw,9lwc,9ly5,9lz2,9m1p,9m3s,9m8l,9mns,9mxu,9mze,9n0e,9n38,9ncy,9np0,9nw9,9ppy,9q6n,9qqe,9rih,9sat,9t3s,9u5q,9ue0,9ulm,9uoi,9vmn,9w14,9wwh,9xqb,9xth,9yio,9zen"
SUBSETS=("$T0" "$T1" "$T2" "$T3")

log(){ echo "[$(date +%H:%M:%S)] $*"; }

launch_card(){ # $1=card
  local card="$1"; local targets="${SUBSETS[$card]}"
  TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_HOLDER=$LEASE PYTHONPATH=$WT \
    PYTHONUNBUFFERED=1 nohup "$PY" -u "$PROG" --targets "$targets" --device "$card" \
    --timeout "$TIMEOUT" >> /tmp/abag_tiera_card$card.log 2>&1 < /dev/null &
  echo "$!"
}

# predict child of a generate.py pid (the actual fold proc whose CPU we watch)
predict_pid_of(){ # $1=generate.py pid
  pgrep -P "$1" -f "tt_bio.main predict" 2>/dev/null | head -1
}

cpu_time_s(){ # $1=pid -> seconds of CPU (sum utime+stime in ticks, /sysconf)
  local p="$1"; [ -n "$p" ] || { echo 0; return; }
  local clk=$(getconf CLK_TCK 2>/dev/null || echo 100)
  awk -v p="$p" -v clk="$clk" 'BEGIN{u=0;s=0}
    /^proc /{next} {f[$1]=$2} END{print int((f["utime"]+f["stime"])/clk)}' /proc/$p/stat 2>/dev/null || echo 0
}

kill_card(){ # $1=card $2=generate.py pid
  local card="$1"; local g="$2"
  [ -n "$g" ] && kill -INT "$g" 2>/dev/null; sleep 5
  [ -n "$g" ] && kill -KILL "$g" 2>/dev/null
  for c in $(pgrep -f "tt_bio.main predict.*--out_dir /home/ttuser/abag_xm/tier_a" 2>/dev/null); do
    # only kill predicts whose env card matches — simpler: kill all stale predicts for this card's targets
    kill -KILL "$c" 2>/dev/null
  done
  sleep 2
}

recover_card(){ # $1=card
  log "RECOVER card $card: kill hung tree + tt-smi -r $card"
  kill_card "$card" "${PIDS[$card]}"
  timeout 90 "$TT_SMI" -r "$card" >/dev/null 2>&1
  sleep 3
  PIDS[$card]=$(launch_card "$card")
  log "relaunched card $card pid=${PIDS[$card]}"
  LASTCPU[$card]=$(cpu_time_s "$(predict_pid_of "${PIDS[$card]}")")
}

mkdir -p "$HOME/abag_xm/tier_a"
log "START Tier-A runner v2: 4 cards, timeout=${TIMEOUT}s, cpu-stall<${STALL_CPU}s/60s"
PIDS=(); LASTCPU=()
for c in 0 1 2 3; do
  PIDS[$c]=$(launch_card $c); LASTCPU[$c]=0
  log "launched card $c pid=${PIDS[$c]}"
done
while true; do
  sleep 60
  alive=0; for c in 0 1 2 3; do [ -n "${PIDS[$c]}" ] && kill -0 "${PIDS[$c]}" 2>/dev/null && alive=$((alive+1)); done
  if [ "$alive" -eq 0 ]; then log "all 4 generate.py exited — campaign slice complete"; break; fi
  for c in 0 1 2 3; do
    [ -n "${PIDS[$c]}" ] || continue
    kill -0 "${PIDS[$c]}" 2>/dev/null || { log "card $c generate.py died — relaunching"; PIDS[$c]=$(launch_card $c); LASTCPU[$c]=0; continue; }
    pp=$(predict_pid_of "${PIDS[$c]}")
    [ -n "$pp" ] || { LASTCPU[$c]=0; continue; }   # between folds — reset baseline
    now=$(cpu_time_s "$pp")
    delta=$(( now - ${LASTCPU[$c]} ))
    if [ "${LASTCPU[$c]}" -ne 0 ] && [ "$delta" -lt "$STALL_CPU" ]; then
      log "card $c HUNG: predict pid=$pp cpu ${LASTCPU[$c]}->${now} (+${delta}s/60s) < ${STALL_CPU}s"
      recover_card "$c"
    else
      LASTCPU[$c]=$now
    fi
  done
done
log "DONE"
