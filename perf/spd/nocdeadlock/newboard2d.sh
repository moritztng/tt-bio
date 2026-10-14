#!/bin/bash
# Phase 2d (2026-10-14, spd-nocdeadlock2). Replaces newboard2/2b/2c. Each chip runs a list of tree:arm:drain legs
# (c730, 20 folds each) with live Inspector RPC (port 5005<chip>) and a 300 s Metal op timeout that runs hangcap.sh
# (tt-triage) against the live hung process. drain=1 compiles every kernel (stock ttnn too) through rtroot/, the wheel's
# tt_metal with tt-metal PR #59896 applied. Tree fix2d = fix2 + tt_bio/metal_overlay.py noc_exit_drain (same patch, on
# at import with the DRAM read split); fix2 = split only. Every leg here uses drain=0 (tt-bio applies its own overlay).
cd ~/spd/spd-nocdeadlock; source ~/hwprobe/env.sh; source ~/tt-bio-dev/env/bin/activate
until grep -q NB-DONE newboard.log; do
  grep -q "c2 mcast x2000" nb_c2.out && pkill -TERM -f "python mcast.py"
  sleep 20; done
exec 8>~/spd_qb2_chip2.lock; flock 8; exec 9>~/spd_qb2_chip3.lock; flock 9
timeout 300 ~/.local/bin/tt-smi -r 2,3; echo "reset rc=$? $(date -u +%FT%TZ)"; sleep 15
run() { c=$1; shift
  for leg in "$@"; do IFS=: read tree a drain <<< "$leg"; arm=$a; [ $a = fast ] && arm=fast:fast
    tag=nb2d_${tree}_d${drain}; lp=~/spd/spd-nocdeadlock/wd/insp_${tag}_c${c}_$a; mkdir -p $lp
    out=~/spd/spd-nocdeadlock/wd/${tag}_c${c}_$a; rm -rf $out
    dr=(); [ $drain = 1 ] && dr=(TT_METAL_RUNTIME_ROOT=$HOME/spd/spd-nocdeadlock/rtroot TT_METAL_CACHE=$HOME/spd/spd-nocdeadlock/cache_drain)
    echo "== c$c $tree $a drain=$drain x20 start $(date -u +%FT%TZ) log=${tag}_c${c}_$a.log"
    (cd $tree && exec env "${dr[@]}" PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$c TT_METAL_LOGS_PATH=$lp TT_METAL_INSPECTOR=1 \
      TT_METAL_INSPECTOR_RPC=1 TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=localhost:5005$c \
      TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=300 \
      TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="/home/ttuser/spd/spd-nocdeadlock/hangcap.sh $c $lp ${tag}_$a 5005$c" \
      timeout -s TERM 4200 python perf/spd/bench.py --out $out --chip $c --arm $arm --inputs c730 --warm 19 > ../${tag}_c${c}_$a.log 2>&1) &
    p=$!
    while kill -0 $p 2>/dev/null; do sleep 15
      grep -q "\"err\": \"" $out/bench.jsonl 2>/dev/null && { echo "c$c errored rep, stopping $(date -u +%FT%TZ)"; pkill -TERM -P $p; pkill -TERM -f -- "--out $out "; sleep 60; }
    done; wait $p; rc=$?
    echo "== c$c $a rc=$rc end $(date -u +%FT%TZ) ok_reps=$(grep -c "\"err\": null" $out/bench.jsonl)"
    if grep -q "\"err\": \"" $out/bench.jsonl || [ $rc -ne 0 ]; then echo "c$c HANG leg=$leg"
      # the board pair must be reset before this chip runs again; the sibling chip may still be busy, so stop here
      return; fi
  done
}
run 3 fix2d:exact:0 fix2d:fast:0 fix2:exact:0 > nb2d_c3.out 2>&1 &
run 2 fix2d:exact:0 fix2d:fast:0 > nb2d_c2.out 2>&1 &
wait; echo "NB2D-DONE $(date -u +%FT%TZ)"
