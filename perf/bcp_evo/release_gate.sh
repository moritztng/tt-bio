#!/bin/bash
# Full release gate on the bcp-evo merge tree, qb2 card 3. Log perf/bcp_evo/out/gate/release_gate.txt.
cd ~/.coworker/wt/bcp-evo
O=perf/bcp_evo/out/gate
echo "gate start $(date -u +%FT%TZ) $(git rev-parse HEAD)" >> $O/chain.txt
PYTHONPATH=$PWD TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcp-evo \
  timeout 36000 ~/bcx_e2e_venv/bin/python3 scripts/release_gate.py > $O/release_gate.txt 2>&1
echo "gate rc=$? $(date -u +%FT%TZ)" >> $O/chain.txt
