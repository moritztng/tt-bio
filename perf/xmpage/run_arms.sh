#!/usr/bin/env bash
# The page protocol (scripts/gpu_vs_tt/tt_baseline.py, 512 aa cdk2x2, batch 1, recycles 10,
# sampling_steps 200) re-run on main after the accurate-softmax merge, both arms on one card.
# ON  = shipped main defaults (accurate softmax on the Protenix-v2 / OpenDDE Pairformer sites)
# OFF = TT_BIO_ACCURATE_SOFTMAX_AB=-all, i.e. what main did before cd20c2f1
# One card, serialized under benchlock: a co-tenanted timed run is a wrong number, not a noisy one.
set -u
source /home/ttuser/.coworker/wt/protenix-opendde-softmax-perfpage-remeasure/perf/xmpage/env.sh
cd $WT || exit 1
arm_run () {   # model arm tag
    local m=$1 arm=$2 tag=$3
    echo "=== $m arm=$arm tag=$tag $(date -Is) ==="
    if [ "$arm" = off ]; then export TT_BIO_ACCURATE_SOFTMAX_AB=-all; else unset TT_BIO_ACCURATE_SOFTMAX_AB; fi
    $BL protenix-opendde-softmax-perfpage-remeasure -- $PY -u scripts/gpu_vs_tt/tt_baseline.py \
        --model $m --repeat 3 \
        --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
        --label "512 aa cdk2x2, page protocol, qb1 card 1, softmax $arm" \
        --msa-dir $WT/.msa_xmpage \
        --keep-cif $O/cif_${tag} --out $O/${tag}.json
    echo "RC=$? $(date -Is)"
}
arm_run protenix-v2 on  px_on_c1
arm_run protenix-v2 off px_off_c1
arm_run opendde     on  od_on_c1
arm_run opendde     off od_off_c1
arm_run protenix-v2 on  px_on2_c1
arm_run opendde     on  od_on2_c1
echo "ALL DONE $(date -Is)"
