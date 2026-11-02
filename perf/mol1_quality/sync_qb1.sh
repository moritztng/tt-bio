#!/bin/bash
# Runs on qb2 beside a TT-BH arm: every 5 min, finished folds move between qb2:~/mol1-quality/OUT and qb1's share
# (bh_arm.sh OUT on qb1, log ~/mol1-quality/OUT.log there) so neither box refolds a case+seed the other finished. qb2's
# rc markers go to qb1; each qb1 fold with rc=0 that qb2 lacks comes back whole (.runs/<case>/seed_<s> +
# mol1_results_<case>/<case>/seed_<s>). qb1's AICLK record lands as OUT/aiclk-qb1.log. Ends after qb1 logs ARM-EXIT.
#   bash sync_qb1.sh OUT     log on stdout, SYNC-EXIT at the end
set -uo pipefail
OUT=${1:?OUT}; mkdir -p ~/mol1-quality/$OUT; cd ~/mol1-quality/$OUT; Q=ttuser@tt-quietbox; say() { echo "$(date -u +%FT%TZ) $*"; }
while :; do
  fin=$(timeout 60 ssh -o BatchMode=yes $Q "grep -c ARM-EXIT ~/mol1-quality/$OUT.log" 2>/dev/null)
  find .runs -name rc | xargs grep -l rc=0 2>/dev/null > /tmp/mq-qb2-$OUT.lst
  timeout 600 rsync -a --ignore-existing --files-from=/tmp/mq-qb2-$OUT.lst . $Q:mol1-quality/$OUT/ 2>/dev/null
  timeout 120 ssh -o BatchMode=yes $Q "cd ~/mol1-quality/$OUT 2>/dev/null && find .runs -name rc | xargs grep -l rc=0 2>/dev/null" \
    > /tmp/mq-qb1-$OUT.lst
  n=0
  while read -r rc; do
    [ -f "$rc" ] && continue
    d=${rc%/rc}; s=${d##*/}; c=${d#.runs/}; c=${c%/*}
    r=mol1_results_${c%%.json*}/$c/$s  # a JSON-file case keeps .json (and a multi-entry file its _<entry>) in its name, not in its results dir
    timeout 1800 rsync -a --exclude claim $Q:mol1-quality/$OUT/$r/ $r/ --mkpath &&
      timeout 300 rsync -a --exclude claim $Q:mol1-quality/$OUT/$d/ $d/ --mkpath && echo qb1 > $d/host && n=$((n + 1))
  done < /tmp/mq-qb1-$OUT.lst
  timeout 600 rsync -a $Q:mol1-quality/$OUT/aiclk.log aiclk-qb1.log 2>/dev/null
  say "synced $n qb1 folds back; qb1 has $(wc -l < /tmp/mq-qb1-$OUT.lst) done"
  [ "${fin:-0}" -ge 1 ] && { say "SYNC-EXIT"; exit 0; }
  sleep 300
done
