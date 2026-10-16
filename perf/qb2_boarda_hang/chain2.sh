#!/bin/bash
# chain2.sh: after chain.sh (r1B, r1A) finishes, rerun board B as r2B with the NoC progress watch (r1B was frozen by
# tt-triage halting its BRISCs, so it says nothing about board B).
W=$HOME/qb2-boarda-hang; cd $W
until grep -q CHAIN-DONE $W/chain.log 2>/dev/null; do [ -e $W/STOP ] && exit 0; sleep 60; done
./hk2.sh B r2B; echo "B2 exit $?" >> $W/chain.log
echo "CHAIN2-DONE $(date -u +%FT%TZ)" >> $W/chain.log
