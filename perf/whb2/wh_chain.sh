#!/bin/bash
# One chain so the byte model and the stage split cannot contaminate each other. Graph capture
# inflates every wall it touches, so the byte runs are not measurements, but they still take a card
# and read as a foreign fold to the other workers. Everything goes under the shared benchlock.
#
# Order is deliberate: the 512 split first, because it carries the A/A floor that every delta in
# this task is judged against, then the 512 byte model, then the two 1024 blocks. If the queue eats
# the pass, the most valuable measurement is the one already done.
#
# LOAD_WAIT is 120 s, not 600. Four fleet workers share this box and at least two run their timed
# jobs OUTSIDE the lock (wh-perf-embed-models sets BENCHLOCK_FOREIGN_RE=__never_match__,
# wh-perf-opendde takes a card unlocked), so the quiet condition can never be met -- waiting for it
# only holds the flock for ten minutes and blocks everyone, then proceeds warned anyway. The flock
# still serialises what it can. Co-tenancy is recorded instead: xmodel_ab.py logs loadavg with every
# fold, and the baseAA arm prices what that contention is actually worth.
set -u
W=/home/cust-team/mthuening/whbase
T=$W/tt-bio
export BENCHLOCK_FILE=$W/benchlock
export BENCHLOCK_MAXLOAD=12.0
export BENCHLOCK_LOAD_WAIT_S=120
export BENCHLOCK_WAIT_S=14400
export BENCHLOCK_FOREIGN_RE='whbase/|wt-esmfold2'
. $W/pick_card.sh
LOG=$W/out/split/chain.log
mkdir -p "$W/out/split" "$W/out/bytes"
cd "$T" || exit 1

bytes() {  # size
  local S=$1
  echo "=== bytes $S $(date -u +%H:%M:%S)"
  bash "$W/benchlock.sh" "wh-perf-boltz2-bytes$S" -- \
    bash -c '. /home/cust-team/mthuening/whbase/pick_card.sh; C=$(pick_card) || exit 70; echo "bytes on UMD $C" >&2; exec env TT_VISIBLE_DEVICES=$C "$@"' _ \
    env TT_METAL_LOGGER_LEVEL=FATAL \
        TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 HF_HUB_CACHE=$W/hfcache \
    ./env/bin/python perf/whb2/wh_bytes.py --tree "$T" --size $S --recycles 3 --steps 4 \
      --out "$W/out/bytes/b2_$S.json" > "$W/out/bytes/b2_$S.log" 2>&1
  echo "bytes $S rc=$?"
  tail -4 "$W/out/bytes/b2_$S.log"
}

{
  echo "chain start $(date -u -Is), card picked per job"
  bash "$W/benchlock.sh" wh-perf-boltz2-split512 -- bash "$W/wh_split.sh" 512
  echo "512 block rc=$?"
  bytes 512
  bash "$W/benchlock.sh" wh-perf-boltz2-split1024 -- bash "$W/wh_split.sh" 1024
  echo "1024 block rc=$?"
  bytes 1024
  echo "chain done $(date -u -Is)"
} >> "$LOG" 2>&1
