#!/bin/bash
# spd-bheth on qb2 p300c chip CHIP (lent): ETH-dispatch probe (bare ttnn Tensix then ETH; tt-bio open), then c730
# Tensix vs ETH dispatch, normal (and fast if MODES=both), 1 cold + 3 warm, ttnn 0.68.0+bh.eth2 in a private 3.10 venv.
#   CHIP=1 LOCK=~/spd_qb2_chip1.lock [MODES=both] [DONE=file to touch at exit] bheth_qb2.sh
set -u
B=~/spd-bheth; R=$B/runs/p300c-$(date -u +%m%dT%H%M); mkdir -p $R; PY=$B/venv/bin/python
[ -n "${DONE:-}" ] && trap "touch $DONE; echo \$(date -u +%FT%TZ) touched $DONE >> $R/queue.log" EXIT
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a $R/queue.log; }
cd $B/treeb; export PYTHONPATH=$PWD TT_BIO_LEASE_HOLDER=spd-bheth TT_BIO_LEASE_DIR=$HOME/spd/spd-bheth/leases
mkdir -p $TT_BIO_LEASE_DIR
say "engine $(git rev-parse --short HEAD), ttnn $($PY -c "import importlib.metadata as m; print(m.version(\"ttnn\"))")"
$PY -c "from tt_bio import metal_overlay as M; assert M.bh_eth_dispatch_supported()" || { say "capability check failed"; exit 1; }
exec 9>$LOCK; flock -w ${WAIT:-600} 9 || { say "flock busy"; exit 3; }
export TT_VISIBLE_DEVICES=$CHIP; say "lock held: qb2 chip $CHIP, load $(cut -d" " -f1-3 /proc/loadavg)"
for m in tensix eth; do
  timeout -s INT 300 $PY perf/spd_bh/eth_probe.py $m > $R/probe_$m.log 2>&1; say "probe $m rc=$? $(grep RESULT $R/probe_$m.log)"
done
timeout -s INT 300 $PY -c "
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
from tt_bio import tenstorrent as T
d = T.get_device(); g = d.compute_with_storage_grid_size()
print(\"RESULT ttbio eth_dispatch=%s device_grid=%dx%d main_grid=%s\" % (T.bh_eth_dispatch(), g.x, g.y, T.COMPUTE_GRID_MAIN))
T.cleanup()" > $R/probe_ttbio.log 2>&1; say "probe ttbio rc=$? $(grep RESULT $R/probe_ttbio.log)"
grep -q "mode=eth .*matmul_finite=True" $R/probe_eth.log || { say "ETH dispatch not usable, stopping"; exit 3; }
ARMS="tensix:TT_BIO_BH_ETH_DISPATCH=0 eth"; [ "${MODES:-}" = both ] && ARMS="$ARMS fasttensix:TT_BIO_BH_ETH_DISPATCH=0:fast fasteth:fast"
for ARM in $ARMS; do A=${ARM%%:*}
  timeout -s INT 1500 $PY perf/spd/bench.py --out $R/$A --chip $CHIP --arm "$ARM" --inputs c730 --warm 3 > $R/$A.log 2>&1
  say "arm $A rc=$? $(grep -c "\"ev\": \"rep\"" $R/$A.log) reps"
done
say "end"
