#!/bin/bash
# rec_check.py on one card: rec1.sh <card>
wt=$(cd "$(dirname "$0")/../../.." && pwd); cd "$wt"
export PYTHONPATH=$wt:$wt/perf TT_VISIBLE_DEVICES=$1 TT_BIO_LEASE_CARDS=$1 TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tok512
echo "start $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"
timeout 900 /home/ttuser/tt-bio-dev/env/bin/python perf/rfd4/tok512/rec_check.py --out perf/rfd4/tok512/rec1/rec.json
echo "EXIT $? $(date -u +%FT%TZ)"
