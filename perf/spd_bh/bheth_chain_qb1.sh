#!/bin/bash
# spd-bheth on qb1: wait for the ttnn 0.68.0+bh.eth2 wheel, build a private venv, check it on the CPU, then queue on one
# p150a card by its flock and run: dispatch probe (bare ttnn, Tensix then ETH; tt-bio's own open), then
# c730 Tensix vs ETH dispatch in normal and fast mode, 1 cold + 3 warm each.
#   [QB1_CARD=N] chain_qb1.sh ENGINE_SHA
set -u
SHA=${1:?engine sha}
B=~/spd-bheth; R=$B/runs/p150a-$(date -u +%m%dT%H%M); mkdir -p $R
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a $R/queue.log; }
node_of(){ case $1 in 0) echo 1;; 1) echo 2;; 2) echo 3;; 3) echo 0;; esac; }   # qb1 logical -> /dev/tenstorrent node

while pgrep -f "^/bin/bash [^ ]*build_wheel.sh $B/w1" >/dev/null; do sleep 120; done   # anchored: a shell quoting this command must not match
W=$(ls $B/w1/dist/ttnn-0.68.0+bh.eth2-*.whl 2>/dev/null | head -1)
[ -n "$W" ] || { say "no wheel (see $B/build.log)"; exit 1; }
say "wheel $(basename $W) sha256 $(sha256sum $W | cut -c1-16)"

[ -x $B/venv/bin/python ] || cp -a ~/tt-bio-dev/env $B/venv || { say "venv failed"; exit 1; }
PY=$B/venv/bin/python
if [ "$($PY -c 'import importlib.metadata as m; print(m.version("ttnn"))')" != 0.68.0+bh.eth2 ]; then
  $PY -m pip install -q --no-deps --force-reinstall "$W" || { say "wheel install failed"; exit 1; }
  # tt-bio's runtime-root overlays of the old wheel carry its JIT cache; drop them so nothing links against it.
  for o in ~/.cache/tt_bio/metal_overlay/*/; do
    case "$(readlink $o/build)" in $B/venv/*) rm -rf "$o";; esac
  done
fi
[ -d $B/tree ] || git clone -q /home/ttuser/.coworker/wt/spd-bheth $B/tree
git -C $B/tree fetch -q origin && git -C $B/tree checkout -q $SHA || { say "no engine $SHA"; exit 1; }
cd $B/tree; export PYTHONPATH=$PWD TT_BIO_LEASE_HOLDER=spd-bheth
say "engine $(git rev-parse --short HEAD), ttnn $($PY -c 'import importlib.metadata as m; print(m.version("ttnn"))')"
$PY -c "from tt_bio import metal_overlay as M; assert M.bh_eth_dispatch_supported(), 'capability check says no'" \
  || { say "capability check failed"; exit 1; }

# Block on one card like every other row: a 60 s flock -n poll never wins against flock -w waiters.
CARD=${QB1_CARD:-2}; exec 9>~/spd_qb1_card$CARD.lock
flock -w 43200 9 || { say "no qb1 card $CARD in 12 h"; exit 1; }
N=$(node_of $CARD); export TT_VISIBLE_DEVICES=$CARD
dead(){ [ "$(cat /sys/class/tenstorrent/tenstorrent!$N/tt_heartbeat 2>/dev/null)" = 4294967295 ]; }
say "lock held: qb1 card $CARD (node $N), load $(cut -d' ' -f1-3 /proc/loadavg)"

for m in tensix eth; do
  timeout -s INT 300 $PY perf/spd_bh/eth_probe.py $m > $R/probe_$m.log 2>&1; say "probe $m rc=$? $(grep RESULT $R/probe_$m.log)"
  dead && { say "node $N ARC dead after probe $m, stopping"; exit 2; }
done
timeout -s INT 300 $PY -c "
from tt_bio import tenstorrent as T
d = T.get_device(); g = d.compute_with_storage_grid_size()
print('RESULT ttbio eth_dispatch=%s device_grid=%dx%d main_grid=%s' % (T.bh_eth_dispatch(), g.x, g.y, T.COMPUTE_GRID_MAIN))
T.cleanup()" > $R/probe_ttbio.log 2>&1; say "probe ttbio rc=$? $(grep RESULT $R/probe_ttbio.log)"
grep -q "mode=eth .*grid=12x10.*matmul_finite=True" $R/probe_eth.log || { say "ETH dispatch not usable, stopping"; exit 3; }

for ARM in tensix:TT_BIO_BH_ETH_DISPATCH=0 eth fasttensix:TT_BIO_BH_ETH_DISPATCH=0:fast fasteth:fast; do A=${ARM%%:*}
  nice -n 5 timeout -s INT 3600 $PY perf/spd/bench.py --out $R/$A --chip $CARD --arm "$ARM" --inputs c730 --warm 3 > $R/$A.log 2>&1
  say "arm $A rc=$? $(grep -c '"ev": "rep"' $R/$A.log) reps"
  dead && { say "node $N ARC dead after $A, stopping"; exit 2; }
done
say "end"
