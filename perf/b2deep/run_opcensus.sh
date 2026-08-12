#!/bin/bash
# Op-level census of the 512 aa boltz-2 fold. One process, one cold fold, one counted warm fold.
set -u
cd /home/ttuser/.coworker/wt/boltz2-512aa-deep-perf
exec /home/ttuser/.coworker/scripts/benchlock.sh boltz2-512aa-deep-perf -- \
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-512aa-deep-perf \
      PYTHONPATH=/home/ttuser/.coworker/wt/boltz2-512aa-deep-perf \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2deep/opcensus.py \
      --size 512 --out perf/b2deep/opcensus_512.json
