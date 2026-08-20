#!/usr/bin/env bash
# One page-cell run: benchlock, one card, one arm tag. Args: <card> <tag>
set -u
cd /home/ttuser/.coworker/wt/rfd3-page-remeasure-l2-l5a-land
CARD="$1"; TAG="$2"
exec ~/.coworker/scripts/benchlock.sh worker:rfd3-page-remeasure-l2-l5a-land -- \
  env TT_VISIBLE_DEVICES="$CARD" \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-remeasure-l2-l5a-land \
      PYTHONPATH=/home/ttuser/.coworker/wt/rfd3-page-remeasure-l2-l5a-land \
      RFD3_CARD="$CARD" RFD3_E2E_NWARM=3 \
      RFD3_E2E_OUT=perf/p78/results/page_cell.jsonl \
      RFD3_HOST=qb2 RFD3_TTNN=0.68.0 \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/p78/page_cell.py R4 1 "$TAG"
