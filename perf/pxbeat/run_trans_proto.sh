#!/usr/bin/env bash
WT=/home/ttuser/.coworker/wt/protenix-v2-beat-dgx-h200
cd "$WT" || exit 1
BENCHLOCK_WAIT_S=1200 /home/ttuser/.coworker/scripts/benchlock.sh worker:protenix-v2-beat-dgx-h200 -- \
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-v2-beat-dgx-h200 PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/pxbeat/trans_f1_proto.py \
  --out perf/pxbeat/trans_f1_proto_512_c1.json
