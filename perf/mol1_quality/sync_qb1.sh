#!/bin/bash
# Runs on qb2 beside the TT-BH grade: every 5 min, finished folds move between qb2:~/mol1-quality/tt-bh and qb1's share
# (qb1_fanout.sh) so neither box refolds a case+seed the other finished. qb2's rc markers go to qb1; each qb1 fold
# with rc=0 that qb2 lacks comes back whole (.runs/<case>/seed_<s> + mol1_results_<case>/<case>/seed_<s>).
# qb1's AICLK record lands as tt-bh/aiclk-qb1.log. Ends after qb1 logs FANOUT-EXIT with no qb1_extend.sh pending (EXTEND). Log ~/mol1-quality/sync.log.
set -uo pipefail
cd ~/mol1-quality/tt-bh; Q=ttuser@tt-quietbox; say() { echo "$(date -u +%FT%TZ) $*"; }
while :; do
  fin=$(timeout 60 ssh -o BatchMode=yes $Q '[ -e ~/mol1-quality/EXTEND ] || grep -c FANOUT-EXIT ~/mol1-quality/fanout.log' 2>/dev/null)
  find .runs -name rc | xargs grep -l rc=0 2>/dev/null > /tmp/mq-qb2-rc.lst
  timeout 600 rsync -a --ignore-existing --files-from=/tmp/mq-qb2-rc.lst . $Q:mol1-quality/tt-bh/ 2>/dev/null
  timeout 120 ssh -o BatchMode=yes $Q 'cd ~/mol1-quality/tt-bh 2>/dev/null && find .runs -name rc | xargs grep -l rc=0 2>/dev/null' \
    > /tmp/mq-qb1-rc.lst
  n=0
  while read -r rc; do
    [ -f "$rc" ] && continue
    d=${rc%/rc}; s=${d##*/}; c=${d#.runs/}; c=${c%/*}
    r=mol1_results_${c%.json}/$c/$s  # a JSON-file case keeps .json in its name but not in its results dir
    timeout 1800 rsync -a --exclude claim $Q:mol1-quality/tt-bh/$r/ $r/ --mkpath &&
      timeout 300 rsync -a --exclude claim $Q:mol1-quality/tt-bh/$d/ $d/ --mkpath && echo qb1 > $d/host && n=$((n + 1))
  done < /tmp/mq-qb1-rc.lst
  timeout 600 rsync -a $Q:mol1-quality/tt-bh/aiclk.log aiclk-qb1.log 2>/dev/null
  say "synced $n qb1 folds back; qb1 has $(wc -l < /tmp/mq-qb1-rc.lst) done"
  [ "${fin:-0}" -ge 1 ] && { say "SYNC-EXIT"; exit 0; }
  sleep 300
done
