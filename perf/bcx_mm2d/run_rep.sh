#!/usr/bin/env bash
# Second timed pass: the fixed probe and an A/B repetition, then the (untimed) gradcheck.
set -u
cd "$(dirname "$0")/../.."
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-mm2d
PY=/home/ttuser/tt-bio-dev/env/bin/python
O=perf/bcx_mm2d
$PY $O/probe.py $O/probe_n256.json
$PY $O/ab.py --out $O/ab_n256_r2.json --n 256 --pairs 20 --no-f64
$PY perf/hallgrad/gradcheck.py --cases linear,linear3d,mm_tb,mm_tb3d,chain,fanin,lora,lora_pair,lora_bias > $O/gradcheck.txt 2>&1
echo "gradcheck rc=$?"
