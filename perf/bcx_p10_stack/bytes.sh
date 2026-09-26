#!/bin/bash
# Has the hifi route moved the ROUND's byte count, or only its overlap?
#
# `bcx-p10-bytes` is funded off this and the reachability argument turns on it: at 2409 GB
# against a 442.3 GB/s roof the device floor is 5.45 s, so a route that cuts seconds without
# cutting bytes does not move the 10x bar. The composed device column went 10.809 -> 9.107 s,
# which is consistent with fewer bytes OR with better overlap, and only a byte count separates
# them.
#
# `perf/bcx_p10_devmap/devmap.py block` is the campaign's byte counter, adopted unchanged: its
# OpTimer totals operand and result bytes per ttnn verb over one Evoformer block. It is the same
# instrument `bcx-p10-triatt` used when it found that route's cut was 1.14x and not 7x.
#
# It CANNOT ride the timed arms -- it wraps every ttnn verb, so it perturbs what a timed round
# measures. This is its own untimed run. Byte counts are deterministic, so the card it runs on
# does not enter the answer.
set -euo pipefail
cd /home/ttuser/.coworker/wt/bcx-p10-stack
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=0,2 TT_BIO_LEASE_HOLDER=worker:bcx-p10-stack
out=perf/bcx_p10_stack/out/bytes
mkdir -p "$out"
run() {
    name=$1; shift
    echo "=== $(date -u +%FT%TZ) $name ===" >> "$out/bytes.log"
    env "$@" /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_devmap/devmap.py block \
        --n 288 --pad 288 --ks 1 --reps 2 --card 2 --out "blk288_stack_$name.json" \
        > "$out/$name.log" 2>&1 || echo "ARM $name exited $?" >> "$out/bytes.log"
}
#run mat  # already taken, blob preserved as out/bytes/blk288_stack_mat.json
#run mat   TT_BIO_TRIATT_TAPED_SDPA=0 TT_BIO_TAPED_KERNELS= TT_BIO_TRIATT_FUSED_HIFI=0 TT_BIO_TRIATT_DIVIDING_K=0
run agtri TT_BIO_TRIATT_TAPED_SDPA=1 TT_BIO_SDPA_OWN_FORWARD=1 TT_BIO_TAPED_KERNELS= TT_BIO_TRIATT_FUSED_HIFI=0
run hifi  TT_BIO_TRIATT_TAPED_SDPA=0 TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi TT_BIO_TRIATT_FUSED_HIFI=1 TT_BIO_TRIATT_DIVIDING_K=1
echo "=== $(date -u +%FT%TZ) bytes done ===" >> "$out/bytes.log"
