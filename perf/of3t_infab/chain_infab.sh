#!/usr/bin/env bash
# of3t-infab on qb2 card 1: the inference A/B of wk/of3t, as pre-registered in PREREGISTERED.md.
# Waits for of3t-paedraws' card 1 chain to exit, exports both trees, then runs infab.py (every
# model, B A B A) and optrace.py (B A B A). Each piece resumes: infab.py skips a finished model,
# optrace skips a finished arm.
#
#   setsid nohup bash perf/of3t_infab/chain_infab.sh [WAIT_PID] > /dev/null 2>&1 &
W=/home/ttuser/.coworker/wt/of3t-infab; P=$W/perf/of3t_infab; S=/tmp/of3t/of3t-infab
BEFORE=c5b346679df0cda6b0aa511a41526b481e8da541; AFTER=589504144e43a3d9dca644c7385f4663c72d4e39
CARD=1; SMI=/home/ttuser/.local/bin/tt-smi; PY=/home/ttuser/tt-bio-dev/env/bin/python3
L=$P/chain.log
mkdir -p $S; cd $W
log() { echo "=== $* $(date -u +%FT%TZ)" >> $L; }

WAIT=${1:-}
if [ -n "$WAIT" ]; then
  log "waiting for pid $WAIT (of3t-paedraws chain on card $CARD) to exit"
  while kill -0 "$WAIT" 2>/dev/null; do sleep 60; done
  log "pid $WAIT gone"
fi
# Nothing else may hold card 1: a device-fd holder whose cwd is not ours is a cotenant.
while true; do
  busy=""
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    ls -l /proc/$p/fd 2>/dev/null | grep -q "/dev/tenstorrent/$CARD\$" && busy="$busy $p"
  done
  [ -z "$busy" ] && break
  log "card $CARD held by$busy, waiting"; sleep 120
done

for side in before after; do
  c=$BEFORE; [ $side = after ] && c=$AFTER
  if [ ! -s $S/$side/.commit ]; then
    rm -rf $S/$side; mkdir -p $S/$side
    git archive $c | tar -x -C $S/$side && echo $c > $S/$side/.commit
    log "exported $side = $c"
  fi
done

python3 perf/c12_orchestrator/pair_guard/host_quiet.py > $P/HOST_QUIET_start.txt 2>&1
log "host_quiet start rc=$? $(tail -1 $P/HOST_QUIET_start.txt)"
log "infab start $(hostname) card $CARD $($SMI -s 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['device_info'][$CARD]['board_info']['board_type'])")"
$PY $P/infab.py --before $S/before --after $S/after --before-commit $BEFORE --after-commit $AFTER \
  --repo $W --inputs $W --card $CARD --workdir $S/work --out $P/INFAB.json >> $P/infab.log 2>&1
log "infab exit $?"

# optrace: TriangleMultiplication's ttnn.graph capture, 32 configs, from each tree.
for i in 0 1 2 3; do
  side=before; [ $((i % 2)) = 1 ] && side=after
  O=$P/OT_${side}$i.json
  [ -s $O ] && continue
  ( cd $S/$side && env -u TT_MESH_GRAPH_DESC_PATH TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:of3t-infab PYTHONPATH=$S/$side OMP_NUM_THREADS=8 \
      timeout 1800 $PY $P/optrace.py $O > $S/ot_${side}$i.log 2>&1 )
  log "optrace $side$i exit $?"
done
python3 perf/c12_orchestrator/pair_guard/host_quiet.py > $P/HOST_QUIET_end.txt 2>&1
log "host_quiet end rc=$? $(tail -1 $P/HOST_QUIET_end.txt)"
log "chain finished"
