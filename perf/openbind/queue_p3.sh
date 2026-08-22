#!/usr/bin/env bash
# Remaining pass-3 device queue on card 1, serialised. Survives a worker restart (setsid nohup).
set -u
WT=/home/ttuser/.coworker/wt/openbind-perf-p3
cd "$WT" || exit 1
L=perf/openbind/tt_results/ab/logs
# wait for the s1_544 sweep already in flight
while pgrep -f "ob_ab.sh 1 ob_apo_544 s1_544" >/dev/null 2>&1; do sleep 20; done
echo "QUEUE: s1_544 clear @ $(date -u +%H:%M:%SZ)"
bash perf/openbind/ob_ab.sh 1 ob_apo_512 s2_512 3 \
     TT_BIO_FP32_SOFTMAX_L1_FLOAT_CORES=0 TT_BIO_FP32_SOFTMAX_L1_FLOAT_CORES=1 > $L/s2_512.log 2>&1
echo "QUEUE: s2_512 done @ $(date -u +%H:%M:%SZ)"
bash perf/openbind/ob_ab.sh 1 ob_apo_1024 s2_1024 2 \
     TT_BIO_FP32_SOFTMAX_L1_FLOAT_CORES=0 TT_BIO_FP32_SOFTMAX_L1_FLOAT_CORES=1 > $L/s2_1024.log 2>&1
echo "QUEUE: s2_1024 done @ $(date -u +%H:%M:%SZ)"
bash perf/openbind/ob_ab.sh 1 ob_apo_512 e6_512 3 \
     TT_BIO_REBLOCK_PERMUTE_GATED=0 TT_BIO_REBLOCK_PERMUTE_GATED=1 > $L/e6_512.log 2>&1
echo "QUEUE COMPLETE @ $(date -u +%H:%M:%SZ)"
