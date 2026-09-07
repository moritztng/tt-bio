#!/bin/bash
# One size-ladder-config protenix-v2 fold. $1=label $2=card $3=rung $4=model ; env WTHR sets the
# W-chunk threshold override (unset = shipped path).
set -u
WT=/home/moritz/.coworker/wt/wh-transition-wchunk-hang-fix
LABEL=$1; CARD=$2; RUNG=$3; MODEL=${4:-protenix-v2}
mkdir -p $WT/perf/wchunk
cd $WT
export PYTHONNOUSERSITE=1 PYTHONPATH=$WT
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:wh-transition-wchunk-hang-fix
export TT_BIO_SIZE_LIMIT=0
[ -n "${WTHR:-}" ] && export TT_BIO_TRANSITION_W_CHUNKING_THRESHOLD=$WTHR
echo "=== $LABEL card=$CARD rung=$RUNG model=$MODEL WTHR=${WTHR:-shipped} start $(date -u +%FT%TZ)"
/home/cust-team/mthuening/tt-bio/env/bin/python3.10 -u -m tt_bio.main predict \
    $WT/perf/size512/fixtures/cdk2x2_${RUNG}.yaml \
    --model $MODEL --single_sequence --sampling_steps 6 --diffusion_samples 1 --seed 0 \
    --out_dir $WT/perf/wchunk/out_$LABEL
echo "=== $LABEL rc=$? end $(date -u +%FT%TZ)"
