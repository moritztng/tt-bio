#!/usr/bin/env bash
# Pass-3 remaining device queue, ONE script so nothing can race another queue.
# The guard is "no ob_ab.sh at all", not "not this cell": part 2's cell-specific guard was
# vacuously true before its cell had started and it launched a second fold on card 1.
set -u
WT=/home/ttuser/.coworker/wt/openbind-perf-p3
cd "$WT" || exit 1
L=perf/openbind/tt_results/ab/logs
wait_clear() {
  while pgrep -f "ob_ab.sh 1 " | grep -qv "^$$\$"; do
    pgrep -f "ob_ab.sh 1 " >/dev/null || break
    sleep 20
  done
}
# wait for the s1_544 sweep already in flight to exit
while pgrep -f "ob_ab.sh 1 ob_apo_544 s1_544" >/dev/null 2>&1; do sleep 20; done
echo "Q3: card 1 clear @ $(date -u +%H:%M:%SZ)"
bash perf/openbind/ob_ab.sh 1 ob_apo_512 e6_512 3 \
     TT_BIO_REBLOCK_PERMUTE_GATED=0 TT_BIO_REBLOCK_PERMUTE_GATED=1 > $L/e6_512.log 2>&1
echo "Q3: e6_512 done @ $(date -u +%H:%M:%SZ)"
bash perf/openbind/ob_ab.sh 1 ob_apo_512 s2_512 3 \
     TT_BIO_FP32_SOFTMAX_L1_FLOAT_CORES=0 TT_BIO_FP32_SOFTMAX_L1_FLOAT_CORES=1 > $L/s2_512.log 2>&1
echo "Q3: s2_512 done @ $(date -u +%H:%M:%SZ)"
bash perf/openbind/ob_ab.sh 1 ob_apo_1024 s2_1024 2 \
     TT_BIO_FP32_SOFTMAX_L1_FLOAT_CORES=0 TT_BIO_FP32_SOFTMAX_L1_FLOAT_CORES=1 > $L/s2_1024.log 2>&1
echo "Q3 COMPLETE @ $(date -u +%H:%M:%SZ)"
