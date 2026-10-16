#!/bin/bash
# hangcap.sh <chip> <out> <why>: dump a hung Protenix-v2 fold while its process is still alive. py-spy (Python, then
# native) of the bench process, then tt-triage per script (core callstacks, running ops, NoC, fast dispatch).
# Run by Metal on a dispatch timeout (TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE) or by run.sh's stall watch.
c=$1; out=$2; why=$3; d=$out/dump; mkdir -p $d
mkdir $d/.taken 2>/dev/null || exit 0
echo "$why $(date -u +%FT%TZ)" > $d/why
T=$HOME/spd/spd-bhfasthang/tvenv/bin; p=$(cat $out/bench.pid)
timeout 120 sudo -n $T/py-spy dump --pid $p > $d/pyspy.txt 2>&1
timeout 300 sudo -n $T/py-spy dump --native --pid $p > $d/pyspy_native.txt 2>&1
V=$(cat $out/venv); S=${V}lib/python3.10/site-packages
export TT_VISIBLE_DEVICES=$c TT_METAL_HOME=$S/ttnn/tt_metal PYTHONPATH=$S/triage:$S
for r in dump_running_operations dump_callstacks dump_aggregated_callstacks check_noc_status dump_fast_dispatch check_arc; do
  timeout -s TERM 600 $T/python $S/triage/triage.py --run=$r -vv --disable-colors --disable-progress --skip-version-check \
    --inspector-rpc-port=5008$c --inspector-log-path=$out/insp/generated/inspector > $d/$r.txt 2>&1
done
echo "$(date -u +%FT%TZ) done" >> $d/why
