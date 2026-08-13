#!/usr/bin/env bash
# Pass-3 leg: does the PM-local L1 memo turn the 512 control back into a no-op?
# The qsplit arm at 512 cost 3.129 s because a persistent-mask CB refusal at the 995-token refiner
# was written into the STOCK wide-q ladder memo, retiring q_chunk=512 for the rest of the fold.
set -u
WT=/home/ttuser/.coworker/wt/opendde-sizes-perf
cd "$WT"
S=perf/sizes/campaign4.status
bash perf/sizes/run_probe.sh 512 on,qsplit,on perf/sizes/qsplitfix_512.json \
    > perf/sizes/qsplitfix_512.log 2>&1
echo "qsplitfix512 rc=$? $(date -Is)" >> "$S"
