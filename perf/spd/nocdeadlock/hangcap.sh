#!/bin/bash
# hangcap.sh <chip> <logs_path> <tag>: tt-triage dump of a hung chip, run by Metal on dispatch timeout
# (TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE) while the host process is still alive.
c=$1; lp=$2; tag=$3; port=${4:-50051}; o=~/spd/spd-nocdeadlock/hang_${tag}_c${c}_$(date -u +%Y%m%dT%H%M%SZ)
S=/home/ttuser/of3t_train_api/stranger/venv/lib/python3.12/site-packages/triage
export TT_VISIBLE_DEVICES=$c TT_METAL_HOME=/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/tt_metal
export PYTHONPATH=$S:/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages
for r in dump_callstacks dump_aggregated_callstacks check_noc_status dump_running_operations dump_fast_dispatch dump_risc_debug_signals; do
  timeout -s TERM 600 ~/triage_venv/bin/python $S/triage.py --run=$r -vv --disable-colors --disable-progress --skip-version-check --inspector-rpc-port=$port --inspector-log-path=$lp/generated/inspector > $o.$r.txt 2>&1
done
timeout -s TERM 600 ~/triage_venv/bin/python ~/spd/spd-nocdeadlock/dump_cb_counters.py -vv --disable-colors --disable-progress --skip-version-check --inspector-rpc-port=$port --inspector-log-path=$lp/generated/inspector > $o.dump_cb_counters.txt 2>&1
timeout -s TERM 900 env PYTHONPATH=$PYTHONPATH:/home/ttuser/spd/spd-nocdeadlock ~/triage_venv/bin/python ~/spd/spd-nocdeadlock/tensix_diff.py -vv --disable-colors --disable-progress --skip-version-check --inspector-rpc-port=$port --inspector-log-path=$lp/generated/inspector > $o.tensix_diff.txt 2>&1
echo "hangcap c$c $tag $(date -u +%FT%TZ) -> $o" >> ~/spd/spd-nocdeadlock/hangcap.log
