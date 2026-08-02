#!/bin/bash
# Close out the embed result-transport fix on the galaxy: does handing back paths instead of
# base64 actually shorten the wall on a real 32-card pool, and are the embeddings identical?
#
# Everything else about that fix is already settled off-hardware -- unit tests, a byte-exact
# single-card run, and the controller leg costed at ~27x through real HTTP. The one number that
# still needs the galaxy is the end-to-end pool wall (predicted 27 s -> ~12-13 s at N=1024).
#
# Production is NOT patched. This runs a private controller and worker pool out of a copy of the
# branch, so ~/mthuening/aiand-bio is never touched; the only production action is the maintenance
# window, which is deployed and restored by the existing tooling.
#
#   bash embed_transport_window.sh <src-dir> [n-workers] [n-seqs]
#   DRY_RUN=1 bash embed_transport_window.sh ...   # run the guards, then stop
#
# DRY_RUN exists because the guards are the part that must not be wrong -- they are what stands
# between this and another worker's folds -- and a guard nobody has ever seen fire is not a guard.
# It exercises them against the live box and exits before touching anything.
#
# Run it detached. It ignores INT/HUP because setsid+nohup alone does not survive the launching
# session going away on this host.
trap "" INT HUP
set -u

SRC=${1:-/home/cust-team/mthuening/g32-src}
NW=${2:-8}
NSEQ=${3:-1024}
DRY_RUN=${DRY_RUN:-0}
M=/home/cust-team/mthuening/maintenance
B=/home/cust-team/mthuening/g32/embedwindow
ENVBIN=/home/cust-team/mthuening/tt-bio/env/bin
LOG=$B/run.log

mkdir -p "$B"
say() { printf '\n=== %s\n' "$*" >> "$LOG"; }
: > "$LOG"

# ---------------------------------------------------------------- guards
say "guards $(date -Is)"
if [ ! -d "$SRC/tt_bio" ]; then echo "ABORT: no branch checkout at $SRC" >> "$LOG"; exit 1; fi

# Someone else's folds must never be interrupted -- that is what caused the 2026-08-01 outage.
if ps -eo args | grep -q "[t]t_bio.main predict"; then
  echo "ABORT: folds are running (another worker owns the box)" >> "$LOG"; exit 1
fi
if [ "$(systemctl is-active japanfold)" = "active" ]; then
  if ! curl -s --max-time 20 http://127.0.0.1:8090/api/cluster \
      | /usr/bin/python3.10 -c 'import sys,json; d=json.load(sys.stdin); sys.exit(0 if not d["runs"] else 1)'; then
    echo "ABORT: the platform has runs in flight" >> "$LOG"; exit 1
  fi
else
  echo "NOTE: japanfold already inactive -- someone else may hold a window; not proceeding" >> "$LOG"
  exit 1        # an inactive unit means someone else took the box, not that it is free
fi
echo "guards passed: box is idle and the platform is serving with nothing in flight" >> "$LOG"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN: would now arm the watchdog, deploy maintenance, start $NW workers from $SRC," >> "$LOG"
  echo "         run the N=$NSEQ A/B, tear the pool down and restore. Stopping here." >> "$LOG"
  exit 0
fi

# ---------------------------------------------------------------- window
say "arm the restore watchdog BEFORE taking anything down"
setsid nohup /home/cust-team/mthuening/g32_restore_watchdog.sh 10 60 >/dev/null 2>&1 < /dev/null &
sleep 2

say "deploy maintenance page"
bash "$M/maint-deploy.sh" >> "$LOG" 2>&1
sleep 20
curl -s -o /dev/null -w "public site during window: %{http_code}\n" -m 25 https://japanfold.com/ >> "$LOG"

# maint-deploy stopping the unit does NOT mean the chips came free: the pool's workers survive the
# stop (leaked multiprocessing children, seen on every batch of this campaign), and maint-deploy
# says so and leaves them. Starting a pool anyway is how the first attempt at this window burned
# ten minutes of maintenance page for nothing -- every worker blocked on device open. Verify the
# box is actually ours, and hand it straight back if it is not.
say "verify the window actually freed the box"
held=$(/usr/bin/python3.10 -c '
import os, glob
h = set()
for p in glob.glob("/proc/[0-9]*/fd/*"):
    try: t = os.readlink(p)
    except OSError: continue
    if "tenstorrent/" in t: h.add(int(t.rsplit("/", 1)[1]))
print(len(h))')
echo "  japanfold=$(systemctl is-active japanfold)  chips still held=$held" >> "$LOG"
if [ "$(systemctl is-active japanfold)" = "active" ] || [ "$held" != "0" ]; then
  echo "ABORT: the window did not free the box ($held chips still held) -- restoring, nothing started" >> "$LOG"
  bash "$M/maint-restore.sh" >> "$LOG" 2>&1
  sleep 60
  curl -s -o /dev/null -w "public site after early restore: %{http_code}\n" -m 30 https://japanfold.com/ >> "$LOG"
  exit 1
fi

# ---------------------------------------------------------------- private pool
say "start a private controller + $NW workers from $SRC"
export PYTHONPATH=$SRC HF_HUB_CACHE=/home/cust-team/mthuening/models TT_METAL_LOGGER_LEVEL=FATAL
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$SRC" || exit 1
setsid nohup $ENVBIN/python -m tt_bio.main controller --listen 127.0.0.1:8899 \
  >> "$B/controller.log" 2>&1 < /dev/null &
sleep 8
for i in $(seq 0 $((NW - 1))); do
  TT_VISIBLE_DEVICES=$i TT_BIO_LEASE_HOLDER=worker:galaxy-32way-scaling-bottleneck \
    setsid nohup $ENVBIN/python -m tt_bio.main worker --connect http://127.0.0.1:8899 \
      >> "$B/worker$i.log" 2>&1 < /dev/null &
done
say "waiting for workers to register"
for _ in $(seq 1 60); do
  n=$(curl -s --max-time 10 http://127.0.0.1:8899/cluster \
      | /usr/bin/python3.10 -c 'import sys,json;print(json.load(sys.stdin).get("online_workers",0))' 2>/dev/null || echo 0)
  echo "  online=$n" >> "$LOG"
  [ "$n" -ge "$NW" ] && break
  sleep 5
done

# ---------------------------------------------------------------- the A/B
say "A/B at N=$NSEQ (cold cell first, discarded -- every worker loads the model once)"
$ENVBIN/python "$SRC/scripts/profiling/embed_transport_ab.py" "$SRC" 64 http://127.0.0.1:8899 \
  >> "$LOG" 2>&1
for rep in 1 2; do
  $ENVBIN/python "$SRC/scripts/profiling/embed_transport_ab.py" "$SRC" "$NSEQ" http://127.0.0.1:8899 \
    >> "$LOG" 2>&1
done

# ---------------------------------------------------------------- teardown
say "stop the private pool (TERM by explicit pid; never KILL -- a killed tt-metal child wedges its chip)"
for p in $(ps -eo pid,args | grep "[t]t_bio.main worker --connect http://127.0.0.1:8899" | awk '{print $1}'); do
  kill -TERM "$p"
done
sleep 30
for p in $(ps -eo pid,args | grep "[t]t_bio.main controller --listen 127.0.0.1:8899" | awk '{print $1}'); do
  kill -TERM "$p"
done
sleep 10
echo "chips still held: $(/usr/bin/python3.10 -c '
import os,glob
h=set()
for p in glob.glob("/proc/[0-9]*/fd/*"):
    try: t=os.readlink(p)
    except OSError: continue
    if "tenstorrent/" in t: h.add(int(t.rsplit("/",1)[1]))
print(sorted(h))')" >> "$LOG"

say "restore production"
bash "$M/maint-restore.sh" >> "$LOG" 2>&1
sleep 90
curl -s -o /dev/null -w "public site after restore: %{http_code}\n" -m 30 https://japanfold.com/ >> "$LOG"
curl -s --max-time 20 http://127.0.0.1:8090/api/cluster >> "$LOG" 2>&1
say "WINDOW_DONE $(date -Is)"
