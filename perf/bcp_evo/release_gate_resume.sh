#!/bin/bash
# Resume the release gate on a qb2 card from a clean detached checkout of the gated sha, so the
# journal PASS records for that commit carry over and nothing runs on the moving branch tree.
# Usage: release_gate_resume.sh <sha> <card>   (card = a fleet-leased card with an empty node; checkout at perf/bcp_evo/out/gatetree, journal copied in)
C=${2:?card}
fuser -s /dev/tenstorrent/$C && { echo "card $C node is open"; exit 1; }
cd ~/.coworker/wt/bcp-evo/perf/bcp_evo/out/gatetree || exit 1
O=~/.coworker/wt/bcp-evo/perf/bcp_evo/out/gate
[ "$(git rev-parse --short=9 HEAD)" = "$1" ] || { echo "gatetree is not at $1"; exit 1; }
echo "gate resume start $(date -u +%FT%TZ) $(git rev-parse HEAD) card $C" >> $O/chain.txt
PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bcp-evo \
  timeout 36000 ~/bcx_e2e_venv/bin/python3 scripts/release_gate.py --resume > $O/release_gate_resume.txt 2>&1
echo "gate resume rc=$? $(date -u +%FT%TZ)" >> $O/chain.txt
