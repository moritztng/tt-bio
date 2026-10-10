#!/bin/bash
# Is Ethernet dispatch run-to-run deterministic? c730 seed 103 folded cold REPS times per arm, each in a fresh process,
# with every diffusion step hashed per sample (steptap.py). On p300c an ETH fold of seed 103 came out in one of two
# digests (sample 4 only); this says whether Tensix dispatch ever does the same and at which step the samples part.
#   CHIP=N LOCK=<flock> [REPS=3] [ARMS="tensix eth"] [DONE=file] determinism.sh   (run from a tree on the eth2 venv)
set -u
B=~/spd-bheth; R=$B/runs/det-$(hostname -s)-c$CHIP-$(date -u +%m%dT%H%M); mkdir -p $R; PY=$B/venv/bin/python
[ -n "${DONE:-}" ] && trap "touch $DONE" EXIT
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a $R/queue.log; }
export PYTHONPATH=$PWD TT_BIO_LEASE_HOLDER=spd-bheth TT_BIO_LEASE_DIR=$HOME/spd/spd-bheth/leases; mkdir -p $TT_BIO_LEASE_DIR
say "engine $(git rev-parse --short HEAD), ttnn $($PY -c "import importlib.metadata as m; print(m.version(\"ttnn\"))")"
exec 9>$LOCK; flock -w ${WAIT:-43200} 9 || { say "flock busy"; exit 3; }
export TT_VISIBLE_DEVICES=$CHIP; say "lock held: chip $CHIP, load $(cut -d" " -f1-3 /proc/loadavg)"
for i in $(seq ${REPS:-3}); do for m in ${ARMS:-tensix eth}; do
  A=${m}$i; ARM=eth; [ $m = tensix ] && ARM=tensix:TT_BIO_BH_ETH_DISPATCH=0
  TT_BIO_TRUNK_TAP=$R/$A.tap timeout -s INT 900 $PY perf/spd_bh/steptap.py --out $R/$A --chip $CHIP --arm "$ARM" \
    --inputs c730 --seed 103 --warm 0 > $R/$A.log 2>&1
  say "$A rc=$? $(grep -o "\"digest\": \"[0-9a-f]*\"" $R/$A/bench.jsonl)"
done; done
say end
