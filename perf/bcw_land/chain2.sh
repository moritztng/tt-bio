#!/bin/bash
# Pass 2 on card $1 at HEAD (dbias merged): card-open suites, abb3 re-record, OF3/Protenix training
# suites on this tree and on .base, a paired size-ladder re-check of the three rows the full gate
# failed at load1 ~10 (stack/base/stack), then the round sitting if load1 <= 4. Log out/chain2/.
cd "$(dirname "$0")/../.."
here=$PWD; card=$1; O=$here/perf/bcw_land/out/chain2; mkdir -p $O
log(){ echo "$* $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $O/chain.txt; }
waitcard(){ while fuser /dev/tenstorrent/$card >/dev/null 2>&1; do sleep 20; done; }
export TT_BIO_LEASE_TIMEOUT=3600
( while :; do echo "$(date -u +%s) $(cat /sys/class/tenstorrent/tenstorrent!$card/tt_aiclk 2>/dev/null) $(cut -d" " -f1 /proc/loadavg)"; sleep 10; done ) > $O/aiclk.txt &
ck=$!; trap "kill $ck 2>/dev/null" EXIT
log "start $(git rev-parse HEAD)"
waitcard; bash perf/bcw_land/suites.sh $card > $O/suites_card$card.txt 2>&1; log "suites done"
waitcard; O=$O/abb3 bash perf/bcw_land/abb3_rerecord.sh $card; log "abb3 done"
TR="tests/test_of3_training_equivalence.py tests/test_exact_training_default.py tests/test_training_tape_rebind.py
 tests/test_training_tape_rebind__apbback.py tests/test_training_opt_in.py tests/test_train_interface.py
 tests/test_protenix_diffusion.py tests/test_protenix_dit_consistent.py tests/test_training_full_weights.py
 tests/test_of3t_step_slot_discovery.py"
for t in stack base; do
  tree=$here; [ $t = base ] && tree=$here/.base
  waitcard
  (cd $tree && PYTHONPATH=$tree TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcw-land \
    timeout 2400 ~/bcx_e2e_venv/bin/python3 -m pytest -q -rs -p no:cacheprovider $TR 2>&1 | tail -60; echo "rc=${PIPESTATUS[0]}") > $O/train_$t.txt 2>&1
  log "train $t done"
done
for t in stack1 base1 stack2; do
  tree=$here; [ ${t%?} = base ] && tree=$here/.base
  waitcard
  (cd $tree && PYTHONPATH=$tree TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcw-land \
    timeout 5400 ~/bcx_e2e_venv/bin/python3 scripts/release_gate.py --model size-ladder --load-ceiling 0 \
    --size-ladder-models boltz2,openbind,nesso1 --size-ladder-rungs 256,512,768) > $O/ladder_$t.txt 2>&1
  log "ladder $t rc=$?"
done
for i in $(seq 1 30); do [ "$(awk "{print (\$1<=4)}" /proc/loadavg)" = 1 ] && break; sleep 60; done
if [ "$(awk "{print (\$1<=4)}" /proc/loadavg)" = 1 ]; then log "round start"; bash perf/bcw_land/round_sit.sh $card 9 round2 > $O/round.txt 2>&1; log "round done"
else log "round skipped: load1 never <= 4 in 30 min"; fi
log "end"
