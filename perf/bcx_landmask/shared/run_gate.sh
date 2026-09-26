#!/bin/bash
# usage: run_gate.sh <tree> <tag> <model...>
# release_gate.py fold arms from <tree> on card 3, with a sysfs AICLK sampler and loadavg alongside.
# Same as bcx-land's run_gate.sh with this row's paths; CIFs are copied out after each arm.

# One place decides what a valid AICLK is: perf/lib/aiclk.sh, mirroring tt_bio.aiclk.
_L=$(cd "$(dirname "$0")" && pwd); . "${_L%/perf/*}/perf/lib/aiclk.sh" || exit 1
tree=$1; tag=$2; shift 2
out=/home/ttuser/.coworker/wt/bcx-landmask/perf/bcx_landmask/shared/$tag
mkdir -p $out
( while true; do echo "$(date -u +%H:%M:%S) aiclk=$(aiclk 0) load=$(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done ) > $out/clock.log &
s=$!
cd $tree
for m in "$@"; do
  RELEASE_GATE_SIZE_WORKDIR=/tmp/bcx-landmask-scratch-$tag PYTHONPATH=$tree TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-landmask \
    timeout 2400 /home/ttuser/tt-bio-dev/env/bin/python scripts/release_gate.py --model $m --keep --load-ceiling 10 \
      --journal $out/journal.jsonl > $out/$m.log 2>&1
  echo "$m rc=$?" >> $out/rc.txt
  mkdir -p $out/$m; cp -r $tree/${m}_results_prot/structures $tree/${m}_results_prot/results.json $out/$m/ 2>/dev/null
done
kill $s
