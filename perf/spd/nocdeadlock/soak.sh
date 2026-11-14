#!/bin/bash
# soak.sh <name> <reset: 2,3|none> <chip>=<tree>:<arm>[,<tree>:<arm>...] ...
# c730 Protenix-v2 soak, 20 folds per leg, chips in parallel, each chip's legs in order. Takes every named chip's
# flock (~/spd_qb2_chip<N>.lock) first. Live Inspector RPC + 300 s Metal op timeout -> hangcap.sh (tt-triage) on a hang;
# a chip stops at its first hung leg. Trees are dirs under ~/spd/spd-nocdeadlock. Index soak_<name>.log,
# per chip soak_<name>_c<N>.out, legs <name>_<tree>_c<N>_<arm>.log, jsonl wd/<name>_<tree>_c<N>_<arm>/bench.jsonl.
name=$1; rs=$2; shift 2
cd ~/spd/spd-nocdeadlock; source ~/hwprobe/env.sh; source ~/tt-bio-dev/env/bin/activate
fd=20; for spec in "$@"; do c=${spec%%=*}; eval "exec $fd>~/spd_qb2_chip$c.lock"; flock $fd; fd=$((fd+1)); done
echo "locks held $(date -u +%FT%TZ)"
[ "$rs" != none ] && { timeout 300 ~/.local/bin/tt-smi -r $rs > /dev/null 2>&1; echo "reset $rs rc=$? $(date -u +%FT%TZ)"; sleep 15; }
run() { c=$1; IFS=, read -ra legs <<< "$2"
  for leg in "${legs[@]}"; do tree=${leg%%:*}; a=${leg#*:}; arm=$a; [ $a = fast ] && arm=fast:fast
    tag=${name}_${tree}; lp=~/spd/spd-nocdeadlock/wd/insp_${tag}_c${c}_$a; mkdir -p $lp
    out=~/spd/spd-nocdeadlock/wd/${tag}_c${c}_$a; rm -rf $out
    echo "== c$c $tree($(cat $tree/.tree_sha 2>/dev/null)) $a x20 start $(date -u +%FT%TZ) log=${tag}_c${c}_$a.log"
    (cd $tree && exec env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$c TT_METAL_LOGS_PATH=$lp TT_METAL_INSPECTOR=1 \
      TT_METAL_INSPECTOR_RPC=1 TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=localhost:5007$c \
      TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=300 \
      TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="/home/ttuser/spd/spd-nocdeadlock/hangcap.sh $c $lp ${tag}_$a 5007$c" \
      timeout -s TERM 5400 python perf/spd/bench.py --out $out --chip $c --arm $arm --inputs c730 --warm 19 > ../${tag}_c${c}_$a.log 2>&1) &
    p=$!
    while kill -0 $p 2>/dev/null; do sleep 15
      grep -q "\"err\": \"" $out/bench.jsonl 2>/dev/null && { echo "c$c errored rep, stopping $(date -u +%FT%TZ)"; pkill -TERM -P $p; pkill -TERM -f -- "--out $out "; sleep 60; }
    done; wait $p; rc=$?
    echo "== c$c $tree $a rc=$rc end $(date -u +%FT%TZ) ok_reps=$(grep -c "\"err\": null" $out/bench.jsonl)"
    if grep -q "\"err\": \"" $out/bench.jsonl || [ $rc -ne 0 ]; then echo "c$c HANG leg=$leg"; return; fi
  done
}
for spec in "$@"; do run ${spec%%=*} ${spec#*=} > soak_${name}_c${spec%%=*}.out 2>&1 & done
wait; echo "SOAK-DONE $name $(date -u +%FT%TZ)"
