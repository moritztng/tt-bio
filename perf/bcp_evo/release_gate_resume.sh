#!/bin/bash
# Resume the release gate on qb2 card 0 from a clean detached checkout of the gated sha, so the
# journal PASS records for that commit carry over and nothing runs on the moving branch tree.
# Usage: release_gate_resume.sh <sha>   (checkout at perf/bcp_evo/out/gatetree, journal copied in)
cd ~/.coworker/wt/bcp-evo/perf/bcp_evo/out/gatetree || exit 1
O=~/.coworker/wt/bcp-evo/perf/bcp_evo/out/gate
[ "$(git rev-parse --short=9 HEAD)" = "$1" ] || { echo "gatetree is not at $1"; exit 1; }
echo "gate resume start $(date -u +%FT%TZ) $(git rev-parse HEAD) card 0" >> $O/chain.txt
PYTHONPATH=$PWD TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcp-evo \
  timeout 36000 ~/bcx_e2e_venv/bin/python3 scripts/release_gate.py --resume > $O/release_gate_resume.txt 2>&1
echo "gate resume rc=$? $(date -u +%FT%TZ)" >> $O/chain.txt
