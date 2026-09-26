#!/bin/bash
# Does the fused backward move the ROUND's bytes, and by how much of the tri-att backward?
#
# The seconds say -1.117 s of device. A byte census is what says whether that is fewer bytes or
# better overlap, and it is the only way to price what the kernel replaces: if it removes the
# backward's whole op sequence and not just its score traffic it takes out more than the byte
# term the campaign's projection charged it.
#
# `perf/bcx_p10_devmap/devmap.py block` is the campaign's byte counter, adopted unchanged, the
# same instrument bcx-p10-stack used for the hifi forward. Its OpTimer wraps every ttnn verb, so
# it CANNOT ride a timed arm; this is its own untimed run. Byte counts are deterministic, so the
# reps agree exactly and the card does not enter the answer.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bcx-p10-tabwire}
out=perf/bcx_p10_tabwire/out/bytes
mkdir -p "$out"
run() {
    name=$1; shift
    echo "=== $(date -u +%FT%TZ) $name ===" >> "$out/bytes.log"
    env "$@" /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_devmap/devmap.py block \
        --n 288 --pad 288 --ks 1 --reps 2 --card 0 --out "blk288_tabwire_$name.json" \
        > "$out/$name.log" 2>&1 || echo "ARM $name exited $?" >> "$out/bytes.log"
}
# Both arms are the composed hifi route. The ONLY difference is the backward kernel, which is
# what makes the families that do not move the control.
run hifi   TT_BIO_TRIATT_TAPED_SDPA=0 TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi \
           TT_BIO_TRIATT_FUSED_HIFI=1 TT_BIO_TRIATT_DIVIDING_K=1 TT_BIO_TRIATT_BW_FUSED=0
run hifibw TT_BIO_TRIATT_TAPED_SDPA=0 TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi \
           TT_BIO_TRIATT_FUSED_HIFI=1 TT_BIO_TRIATT_DIVIDING_K=1 TT_BIO_TRIATT_BW_FUSED=1
echo "=== $(date -u +%FT%TZ) bytes done ===" >> "$out/bytes.log"
