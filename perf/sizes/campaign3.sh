#!/usr/bin/env bash
# Pass-2 leg: the qsplit fold A/B. 768 is the GO from the op-level screen (2.5265x, torch.equal,
# max_abs_diff 0.0); 512 is the control that must come out a byte-identical no-op, because the gate
# already passes there and `q_num_chunks` is 1 so the split expression is unchanged.
set -u
WT=/home/ttuser/.coworker/wt/opendde-sizes-perf
cd "$WT"
S=perf/sizes/campaign3.status
bash perf/sizes/run_probe.sh 768 on,qsplit,on perf/sizes/qsplit_768.json \
    > perf/sizes/qsplit_768.log 2>&1
echo "qsplit768 rc=$? $(date -Is)" >> "$S"
bash perf/sizes/run_probe.sh 512 on,qsplit perf/sizes/qsplit_512.json \
    > perf/sizes/qsplit_512.log 2>&1
echo "qsplit512 rc=$? $(date -Is)" >> "$S"
