#!/bin/bash
# GATE_EXTRA passes extra release_gate.py flags (e.g. --size-ladder-models a,b).
# Release gate for the of3t-p10batch landing: every default arm of scripts/release_gate.py, one
# process per arm, card 1 of qb2, with a sysfs AICLK sampler alongside (TT_VISIBLE_DEVICES=1).
#   gate.sh <tag> [arm...]      (no arms = release_gate.py --list-arms)
set -uo pipefail
cd "$(dirname "$0")/../.."
tree=$PWD; tag=$1; shift
out=$tree/perf/land_of3t_train/$tag; mkdir -p $out
py=/home/ttuser/tt-bio-dev/env/bin/python
arms=("$@"); [ ${#arms[@]} -gt 0 ] || mapfile -t arms < <(TT_VISIBLE_DEVICES= $py scripts/release_gate.py --list-arms)
( while true; do echo "$(date -u +%T) aiclk=$(cat /sys/class/tenstorrent/tenstorrent!1/tt_aiclk 2>/dev/null) load=$(cut -d" " -f1 /proc/loadavg)"; sleep 10; done ) > $out/clock.log &
s=$!
echo "commit $(git rev-parse --short HEAD) start $(date -u +%FT%TZ)" > $out/rc.txt
for m in "${arms[@]}"; do
  t=2400; [ "$m" = size-ladder ] && t=21600; [ "$m" = capacity ] && t=5400
  RELEASE_GATE_SIZE_WORKDIR=/tmp/lot-gate-$tag PYTHONPATH=$tree TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
    TT_BIO_LEASE_HOLDER=worker:land-of3t-train timeout $t $py scripts/release_gate.py --model $m --keep \
    --load-ceiling 10 ${GATE_EXTRA:-} --journal $out/journal.jsonl > $out/$m.log 2>&1
  echo "$m rc=$? $(date -u +%T)" >> $out/rc.txt
done
echo "gate done $(date -u +%FT%TZ)" >> $out/rc.txt
kill $s
