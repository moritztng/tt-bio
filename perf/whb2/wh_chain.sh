#!/bin/bash
# One chain so the byte model and the stage split cannot contaminate each other, and so neither
# contaminates the other two fleet workers sharing this box. Graph capture inflates every wall it
# touches, so the byte runs are not measurements -- but they still burn host CPU and a card, and an
# unlocked fold reads as a foreign fold to anyone waiting for quiet. Everything goes under the lock.
set -u
W=/home/cust-team/mthuening/whbase
T=$W/tt-bio
export BENCHLOCK_FILE=$W/benchlock
export BENCHLOCK_MAXLOAD=9.0
export BENCHLOCK_LOAD_WAIT_S=600
export BENCHLOCK_WAIT_S=10800
export BENCHLOCK_FOREIGN_RE='whbase/|wt-esmfold2'
export CARD=${CARD:-27}
LOG=$W/out/split/chain.log
mkdir -p "$W/out/split" "$W/out/bytes"
cd "$T" || exit 1
{
  echo "chain start $(date -u -Is) card=$CARD"

  for S in 512 1024; do
    echo "=== bytes $S $(date -u +%H:%M:%S)"
    bash "$W/benchlock.sh" "wh-perf-boltz2-bytes$S" -- \
      env TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL \
          TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 HF_HUB_CACHE=$W/hfcache \
      ./env/bin/python perf/whb2/wh_bytes.py --tree "$T" --size $S --recycles 3 --steps 4 \
        --out "$W/out/bytes/b2_$S.json" > "$W/out/bytes/b2_$S.log" 2>&1
    echo "bytes $S rc=$?"
    tail -4 "$W/out/bytes/b2_$S.log"
  done

  bash "$W/benchlock.sh" wh-perf-boltz2-split512  -- bash "$W/wh_split.sh" 512
  echo "512 block rc=$?"
  bash "$W/benchlock.sh" wh-perf-boltz2-split1024 -- bash "$W/wh_split.sh" 1024
  echo "1024 block rc=$?"
  echo "chain done $(date -u -Is)"
} >> "$LOG" 2>&1
