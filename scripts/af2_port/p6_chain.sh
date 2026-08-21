#!/usr/bin/env bash
# pass 6's timed runs, one benchlock hold. Cheap-and-decisive first (the 208 regression A/B is
# 4 minutes and settles item 3), then M4, then the 848 SDPA_WIDE_K A/B.
set -u
WT=/home/ttuser/.coworker/wt/pxdesign-perf-p6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:pxdesign-perf-p6
cd "$WT" || exit 1
mkdir -p perf/pxdesign logs

say() { echo "=== $(date -Is) $* ==="; }

say "1/5 trunk_timing 208, HEAD ce464f0a"
PYTHONPATH="$WT" timeout 900 "$PY" scripts/af2_port/trunk_timing.py --tokens 208 --passes 7 \
    > perf/pxdesign/tt_pxd_p6_trunk208_head.json 2> logs/trunk208_head.err
echo "rc=$?"

say "2/5 trunk_timing 208, pass-13 tree f78c12b8"
cd "$WT/scratch_p13" || exit 1
PYTHONPATH="$WT/scratch_p13" timeout 900 "$PY" scripts/af2_port/trunk_timing.py --tokens 208 --passes 7 \
    > "$WT/perf/pxdesign/tt_pxd_p6_trunk208_p13.json" 2> "$WT/logs/trunk208_p13.err"
echo "rc=$?"
cd "$WT" || exit 1

say "3/5 M4 skip census, 848 tokens"
PYTHONPATH="$WT" timeout 2400 "$PY" scripts/af2_port/skip_census.py --tokens 848 --passes 3 \
    --out perf/pxdesign/tt_pxd_p6_skip_census_848.json > logs/m4_census.out 2>&1
echo "rc=$?"

say "4/5 trunk_timing 848, SDPA_WIDE_K off (shipped default)"
PYTHONPATH="$WT" TT_BIO_SDPA_WIDE_K=0 timeout 1800 "$PY" scripts/af2_port/trunk_timing.py \
    --tokens 848 --passes 3 > perf/pxdesign/tt_pxd_p6_trunk848_widek0.json 2> logs/trunk848_widek0.err
echo "rc=$?"

say "5/5 trunk_timing 848, SDPA_WIDE_K on"
PYTHONPATH="$WT" TT_BIO_SDPA_WIDE_K=1 timeout 1800 "$PY" scripts/af2_port/trunk_timing.py \
    --tokens 848 --passes 3 > perf/pxdesign/tt_pxd_p6_trunk848_widek1.json 2> logs/trunk848_widek1.err
echo "rc=$?"

say "chain done"
