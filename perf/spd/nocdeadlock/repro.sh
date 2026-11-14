#!/bin/bash
# repro.sh <chip> <tag> <tree> [paircond_repro.py args]: one detached repro leg with live Inspector RPC and a
# 120 s Metal op timeout that runs hangcap.sh against the hung process. Log ~/spd/spd-nocdeadlock/rp_<tag>_c<chip>.log
c=$1; tag=$2; tree=$3; shift 3
cd ~/spd/spd-nocdeadlock; source ~/tt-bio-dev/env/bin/activate
lp=~/spd/spd-nocdeadlock/wd/insp_rp_${tag}_c$c; mkdir -p $lp
cd $tree && exec env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$c TT_METAL_LOGS_PATH=$lp TT_METAL_INSPECTOR=1 \
  TT_METAL_INSPECTOR_RPC=1 TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=localhost:5006$c \
  TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=120 \
  TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="/home/ttuser/spd/spd-nocdeadlock/hangcap.sh $c $lp rp_$tag 5006$c" \
  timeout -s TERM 3600 python ~/spd/spd-nocdeadlock/paircond_repro.py "$@" > ~/spd/spd-nocdeadlock/rp_${tag}_c$c.log 2>&1
