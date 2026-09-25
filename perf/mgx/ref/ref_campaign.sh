#!/usr/bin/env bash
# The whole reference set on one box, in phases sized to the card, each phase split across
# every GPU on the box:
#   1. 1024 and 1280 fixtures, three workers per GPU (each fold fits in a third of 141 GB)
#   2. 1536 fixtures, two workers per GPU
#   3. every cell still not ok, one worker per GPU, so an OOM caused by a neighbour is not recorded as
#      the model's own
#   4. protenix-family cells that still OOM in fp32 alone, re-folded in bf16 (recorded as such)
# Detached on the box: setsid nohup bash perf/mgx/ref/ref_campaign.sh [first_phase] > /root/results/campaign.log
# first_phase (default 1) resumes a campaign a box stop cut short. Resume at 2, not 1: phase 1's
# leftovers are mostly neighbour OOMs, and packing them three to a card again repeats the OOM;
# phase 3 folds them alone.
set -u
first=${1:-1}
cd "$(dirname "$0")/../../.."
B=perf/mgx/ref/ref_batch.sh
fx() { python3 -c "import json;print(' '.join(k for k,v in json.load(open('perf/mgx/ref/fixtures/fixtures.json'))['fixtures'].items() if $1))"; }
small=($(fx "v['rung']<1536")); big=($(fx "v['rung']==1536"))
all_models=($(python3 -c "import json;print(' '.join(json.load(open('perf/mgx/ref/plan.json'))['models']))"))
NG=$(nvidia-smi -L | wc -l)
# slice <i> <words...>: every NG-th word starting at i, so each GPU gets its own share
slice() { local i=$1; shift; local a=("$@") o=() k; for ((k=i; k<${#a[@]}; k+=NG)); do o+=("${a[k]}"); done; echo "${o[*]}"; }
W1="opendde opendde-abag protenix-v1"; W2="protenix-v2 boltz2 openfold3 openbind"; W3="esmfold2 esmfold2-fast rf3"
if [ 1 -ge "$first" ]; then
  echo "phase 1 $(date -u +%FT%TZ) on $NG GPU(s)"
  for ((g=0; g<NG; g++)); do f=$(slice $g "${small[@]}")
    CUDA_VISIBLE_DEVICES=$g bash $B "$W1" "$f" "0 1" > /root/results/p1_g${g}_w1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$g bash $B "$W2" "$f" "0 1" > /root/results/p1_g${g}_w2.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$g bash $B "$W3" "$f" "0 1" > /root/results/p1_g${g}_w3.log 2>&1 &
  done; wait
fi
if [ 2 -ge "$first" ]; then
  echo "phase 2 $(date -u +%FT%TZ)"
  # Split models, not fixtures, over the 2*NG workers: ref_batch loops fixtures outermost, so every
  # worker folds 3abq_1536 (the real complex) before the tiled cdk2x2_1536.
  NW=$((2 * NG))
  for ((w=0; w<NW; w++)); do m=(); for ((k=w; k<${#all_models[@]}; k+=NW)); do m+=("${all_models[k]}"); done
    CUDA_VISIBLE_DEVICES=$((w % NG)) bash $B "${m[*]}" "${big[*]}" "0 1" > /root/results/p2_w${w}.log 2>&1 &
  done; wait
fi
if [ 3 -ge "$first" ]; then
  echo "phase 3 $(date -u +%FT%TZ)"
  for ((g=0; g<NG; g++)); do
    CUDA_VISIBLE_DEVICES=$g bash $B "$(slice $g "${all_models[@]}")" all "0 1" > /root/results/p3_g$g.log 2>&1 &
  done; wait
fi
if [ 4 -ge "$first" ]; then
  echo "phase 4 $(date -u +%FT%TZ)"
  P4=(protenix-v1 protenix-v2 opendde opendde-abag)
  for ((g=0; g<NG; g++)); do
    MGX_ONLY=oom MGX_DTYPE=bf16 CUDA_VISIBLE_DEVICES=$g bash $B "$(slice $g "${P4[@]}")" all "0 1" > /root/results/p4_g$g.log 2>&1 &
  done; wait
fi
echo "CAMPAIGN_DONE $(date -u +%FT%TZ)"
