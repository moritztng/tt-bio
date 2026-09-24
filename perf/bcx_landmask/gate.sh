#!/bin/bash
# usage: gate.sh <tree> <name> [tap_gate args...] -- the af2ig-trunk-device arm of <tree> on card 3, scored
# by <tree>'s device_floor.py against its committed record. Host envelope arms come from the cache that
# perf/bcx_land/tap_gate_cached.py keeps (they never touch af2.py's device code). AICLK + loadavg every 10 s.
wt=/home/ttuser/.coworker/wt/bcx-landmask
tree=$1 name=$2; shift 2
out=$wt/perf/bcx_landmask/gate; mkdir -p $out
( while true; do echo "$(date -u +%H:%M:%S) aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk') load=$(cut -d' ' -f1 /proc/loadavg)"; sleep 10; done ) > $out/$name.clock &
s=$!
start=$(date -u +%s)
ENV_CACHE=/home/ttuser/scratch/landmask-envcache PYTHONPATH=$tree TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-landmask \
  /home/ttuser/tt-bio-dev/env/bin/python3 $wt/perf/bcx_land/tap_gate_cached.py \
  --params /home/ttuser/.boltz/af2/params/params_model_1_ptm.npz --stage complex --device "$@" \
  > $out/$name.stdout 2> $out/$name.stderr
rc=$?
kill $s
floor=$(cd $tree && PYTHONPATH=$tree /home/ttuser/tt-bio-dev/env/bin/python3 scripts/af2_port/device_floor.py \
  --report $out/$name.stdout --committed docs/implementation-parity-data/af2ig-trunk-device.json 2>&1 | tail -1)
commit=$(cat $tree/.commit 2>/dev/null || git -C $tree rev-parse HEAD)
echo "rc=$rc wall=$(( $(date -u +%s) - start ))s commit=$commit floor: $floor" > $out/$name.rc
cat $out/$name.rc
