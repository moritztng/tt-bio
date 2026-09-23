#!/usr/bin/env bash
# The whole reference set on one box, in phases sized to the card:
#   1. 1024 and 1280 fixtures, three workers side by side (each fold fits in a third of 141 GB)
#   2. 1536 fixtures, two workers
#   3. every cell still not ok, one worker, so an OOM caused by a neighbour is not recorded as
#      the model's own
#   4. protenix-family cells that still OOM in fp32 alone, re-folded in bf16 (recorded as such)
# Detached on the box: setsid nohup bash perf/mgx/ref/ref_campaign.sh > /root/results/campaign.log
set -u
cd "$(dirname "$0")/../../.."
B=perf/mgx/ref/ref_batch.sh
small=$(python3 -c "import json;print(' '.join(k for k,v in json.load(open('perf/mgx/ref/fixtures/fixtures.json'))['fixtures'].items() if v['rung']<1536))")
big=$(python3 -c "import json;print(' '.join(k for k,v in json.load(open('perf/mgx/ref/fixtures/fixtures.json'))['fixtures'].items() if v['rung']==1536))")
W1="opendde opendde-abag protenix-v1"; W2="protenix-v2 boltz2 openfold3 openbind"; W3="esmfold2 esmfold2-fast rf3"
echo "phase 1 $(date -u +%FT%TZ)"
bash $B "$W1" "$small" "0 1" > /root/results/p1_w1.log 2>&1 &
bash $B "$W2" "$small" "0 1" > /root/results/p1_w2.log 2>&1 &
bash $B "$W3" "$small" "0 1" > /root/results/p1_w3.log 2>&1 &
wait
echo "phase 2 $(date -u +%FT%TZ)"
bash $B "$W1 $W3" "$big" "0 1" > /root/results/p2_w1.log 2>&1 &
bash $B "$W2" "$big" "0 1" > /root/results/p2_w2.log 2>&1 &
wait
echo "phase 3 $(date -u +%FT%TZ)"
bash $B all all "0 1" > /root/results/p3.log 2>&1
echo "phase 4 $(date -u +%FT%TZ)"
MGX_ONLY=oom MGX_DTYPE=bf16 bash $B "protenix-v1 protenix-v2 opendde opendde-abag" all "0 1" > /root/results/p4.log 2>&1
echo "CAMPAIGN_DONE $(date -u +%FT%TZ)"
