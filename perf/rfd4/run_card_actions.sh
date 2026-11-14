#!/usr/bin/env bash
# Card actions 1 and 2 for rfd4-beam, back to back on the one leased card.
#   1. prove the AF2-IG TT arm: one setup, a batch, and the same candidate scored twice
#   2. measure the matched oracle protocol (5 recycles / 200 sampling steps) at 256 and 512
# The card is board 3 in tt-smi's list, which is where the clock is read from; boards 0 and 1
# are other rows' and board 2 is the excluded chip (state/cardblock-qb2-2).
set -u

WT=/home/ttuser/.coworker/wt/rfd4-beam
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PARAMS=/home/ttuser/.boltz/af2/params/params_model_1_ptm.npz

export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:rfd4-beam

cd "$WT" || exit 1

echo "=== [$(date -u +%FT%TZ)] card action 1: AF2-IG TT arm"
timeout 5400 "$PY" perf/rfd4/tt_arm_af2ig.py --rungs 128,256,512 --params "$PARAMS" --dev 0
echo "=== tt arm rc=$?"

echo "=== [$(date -u +%FT%TZ)] card action 2: matched oracle protocol"
timeout 21600 "$PY" perf/rfd4/oracle_cost.py --card 3 --dev 0 \
    --models openfold3,af2ig --rungs 256,512 --reps 3
echo "=== oracle cost rc=$?"

echo "=== [$(date -u +%FT%TZ)] DONE_SCRIPT"
