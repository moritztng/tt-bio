#!/usr/bin/env bash
# of3t-infab AFTER2 (PREREGISTERED.md addendum): B A2 B A2 on six models, then the walk check.
W=/home/ttuser/.coworker/wt/of3t-infab; P=$W/perf/of3t_infab; S=/tmp/of3t/of3t-infab
BEFORE=c5b346679df0cda6b0aa511a41526b481e8da541
CARD=1; PY=/home/ttuser/tt-bio-dev/env/bin/python3; L=$P/chain.log
cd $W; AFTER2=$(git rev-parse f01fa0813)
log() { echo "=== $* $(date -u +%FT%TZ)" >> $L; }
if [ ! -s $S/after2/.commit ]; then
  rm -rf $S/after2; mkdir -p $S/after2
  git archive $AFTER2 | tar -x -C $S/after2 && echo $AFTER2 > $S/after2/.commit
  log "exported after2 = $AFTER2"
fi
python3 perf/c12_orchestrator/pair_guard/host_quiet.py > $P/HOST_QUIET_after2_start.txt 2>&1
log "after2 host_quiet start rc=$? $(uptime)"
$PY $P/infab.py --before $S/before --after $S/after2 --before-commit $BEFORE --after-commit $AFTER2 \
  --repo $W --inputs $W --card $CARD --workdir $S/work2 --models openfold3,openbind,protenix-v2,opendde,rf3,rfd3 \
  --out $P/INFAB2.json >> $P/infab2.log 2>&1
log "infab2 exit $?"
for side in after after2; do
  [ -s $P/WALK_$side.json ] && continue
  ( cd $S/$side && env -u TT_MESH_GRAPH_DESC_PATH TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:of3t-infab PYTHONPATH=$S/$side timeout 1800 $PY $P/walkcheck.py $P/WALK_$side.json \
      > $S/walk_$side.log 2>&1 )
  log "walk $side exit $? $(tail -1 $S/walk_$side.log)"
done
log "after2 chain finished"
