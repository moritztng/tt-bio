#!/usr/bin/env bash
# The three timed readings for bcx-mm2d, under one benchlock hold. Card 1 on qb2.
set -u
cd "$(dirname "$0")/../.."
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-mm2d
PY=/home/ttuser/tt-bio-dev/env/bin/python
O=perf/bcx_mm2d
$PY $O/probe.py $O/probe_n256.json
$PY $O/ab.py --out $O/ab_n256.json --n 256 --pairs 20
$PY perf/hallgrad/census.py --out $O/census_n256_after.json --n 256 --pairs 10
$PY perf/hallgrad/census_table.py $O/census_n256_after.json > $O/census_n256_after_table.md
