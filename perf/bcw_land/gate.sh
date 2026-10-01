#!/bin/bash
# Full release gate on the bcw-land tree, card $1, after the abb3 re-record has ended.
# Log perf/bcw_land/out/gate/release_gate.txt, rc in chain.txt. Re-run with --resume after a kill.
cd "$(dirname "$0")/../.."
card=$1; O=perf/bcw_land/out/gate; mkdir -p $O
until grep -q "^end" perf/bcw_land/out/abb3/run.txt 2>/dev/null; do sleep 30; done
while fuser /dev/tenstorrent/$card >/dev/null 2>&1; do sleep 20; done
echo "gate start $(date -u +%FT%TZ) $(git rev-parse HEAD)" >> $O/chain.txt
( while :; do echo "$(date -u +%s) $(cat /sys/class/tenstorrent/tenstorrent!$card/tt_aiclk 2>/dev/null) $(cut -d' ' -f1 /proc/loadavg)"; sleep 10; done ) > $O/aiclk.txt &
ck=$!; trap "kill $ck 2>/dev/null" EXIT
PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcw-land \
  timeout 36000 ~/bcx_e2e_venv/bin/python3 scripts/release_gate.py --journal $O/journal.jsonl ${GATE_ARGS:-} >> $O/release_gate.txt 2>&1
echo "gate rc=$? $(date -u +%FT%TZ)" >> $O/chain.txt
