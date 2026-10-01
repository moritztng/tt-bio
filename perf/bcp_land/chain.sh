#!/bin/bash
# bcp-land pass 1: the no-flags baseline round on card 0 alone, then the card-free suite and the
# release gate side by side (the gate grades accuracy, not time).
cd "$(dirname "$0")/../.."
o=perf/bcp_land/out; mkdir -p $o
echo "chain start $(date -u +%FT%TZ) head $(git rev-parse --short HEAD)" > $o/chain.log
perf/bcp_land/sit.sh 9 "base1:auto base2:auto base3:auto" >> $o/chain.log 2>&1
echo "baseline done $(date -u +%FT%TZ)" >> $o/chain.log
( PYTHONPATH=$PWD:$HOME/bcx_e2e/bc2 TT_VISIBLE_DEVICES= ~/bcx_e2e_venv/bin/python3 -m pytest tests/ -q -p no:cacheprovider > $o/suite.log 2>&1; echo "suite rc=$? $(date -u +%FT%TZ)" >> $o/chain.log ) &
( PYTHONPATH=$PWD TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcp-land ~/bcx_e2e_venv/bin/python3 scripts/release_gate.py > $o/gate.log 2>&1; echo "gate rc=$? $(date -u +%FT%TZ)" >> $o/chain.log ) &
wait
echo "chain done $(date -u +%FT%TZ)" >> $o/chain.log
