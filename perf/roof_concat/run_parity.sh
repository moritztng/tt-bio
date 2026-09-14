#!/usr/bin/env bash
# The accuracy legs. One process per arm, because the lever changes the SHAPE of proj_g/proj_o and
# so is decided when the module is built, not per call. Retried around the card lease the same way
# the timed legs are: the four cards on this box belong to other workers' chains and turn over.
set -u
WT=/home/ttuser/.coworker/wt/roof-concat-heads-bh
cd "$WT" || exit 1
D=perf/roof_concat/parity
LOG=/tmp/parity_chain.log
mkdir -p "$D"
run() {  # fixture tag arm seed
  local fx=$1 tag=$2 arm=$3 seed=$4
  [ -s "$D/${fx}_${tag}.json" ] && { echo "skip ${fx}_${tag}" >> "$LOG"; return 0; }
  for i in $(seq 1 25); do
    echo "=== ${fx}_${tag} attempt $i $(date -Is) load=$(cut -d' ' -f1 /proc/loadavg)" >> "$LOG"
    env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
        TT_BIO_LEASE_HOLDER=worker:roof-concat-heads-bh PYTHONPATH="$WT" \
      timeout 1200 /home/ttuser/tt-bio-dev/env/bin/python perf/roof_concat/fold_parity.py \
        --dir "$D" --arm "$arm" --tag "${fx}_${tag}" --fixture "$fx" --seed "$seed" >> "$LOG" 2>&1
    [ -s "$D/${fx}_${tag}.json" ] && return 0
    sleep 20
  done
  return 1
}
for fx in cdk2x2_512 cdk2x2_298; do
  run "$fx" off      off 42
  run "$fx" on       on  42
  run "$fx" off2     off 42
  run "$fx" offseed  off 1234
done
env PYTHONPATH="$WT" /home/ttuser/tt-bio-dev/env/bin/python perf/roof_concat/fold_parity.py \
  --dir "$D" --compare >> "$LOG" 2>&1
echo "=== CHAIN DONE $(date -Is)" >> "$LOG"
