#!/bin/bash
# pvo-tt: OpenDDE and Mol-1 seconds per prediction on one qb1 p150a at PopVax-shaped complexes (state/pvo/INPUTS.md timing
# subset 9TH6 9LV4 9W3L, plus 9DBP at 791 tok for the stated ~800 aa), exact and fast as frozen in state/pvo/MODES.md.
# Books all four qb1 cards once at the arbiter's first clear window (perf/mol1/fair/run.py --part p150a discards a round on
# any foreign device holder), holds the four card flocks, then runs every arm from a detached tree at the MODES pin.
#   G=<out dir> D=<data dir: inputs/ + msa/cache> bash perf/pvo_tt/bh.sh
set -u
W=$(cd "$(dirname "$0")/../.." && pwd); PY=/home/ttuser/tt-bio-dev/env/bin/python; C=0; PIN=${PIN:-c52ab1702}
G=${G:?}; D=${D:?}; OWN=pvo-tt-bh; SPAN=${SPAN:-14400}; T0=$G/tree
mkdir -p $G; say() { echo "$(date -u +%FT%TZ) $*" >> $G/bh.log; }
say "start pid $$"
cd /home/ttuser/spd/feed
end() { awk -v o=$OWN '$3 == o {print $2}' /home/ttuser/spd/feed/windows.tsv; }
while :; do
  python3 book.py list | grep -q " $OWN " ||
    python3 book.py add $OWN 0,1,2,3 $SPAN --note "pvo-tt BH OpenDDE+Mol-1 timing (run.py needs the box quiet)" >> $G/book.log
  if python3 book.py active $C | tr " " "\n" | grep -qx $OWN; then
    if [ $(( $(end) - $(date +%s) )) -lt $((SPAN - 1800)) ]; then say "window too short, rebook"; python3 book.py rm $OWN; sleep 60; continue; fi
    got=0; for i in 0 1 2 3; do eval "exec $((5+i))>/home/ttuser/spd_qb1_card$i.lock"; flock -w 300 $((5+i)) && got=$((got+1)) || break; done
    [ $got = 4 ] && break
    for i in 0 1 2 3; do eval "exec $((5+i))>&-"; done
    say "card locks busy, rebook"; python3 book.py rm $OWN
  fi
  sleep 60; done
say "booking active, locks held"
[ -d $T0 ] || git -C $W worktree add -q --detach $T0 $PIN || { say "tree failed"; echo tree > $G/DONE; exit 4; }
say "tree $(git -C $T0 rev-parse HEAD)"
FW=$(~/.local/bin/tt-smi -s 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["device_info"][0]["firmwares"]["fw_bundle_version"])')
say "fw $FW"
run() { TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:pvo-tt PYTHONPATH=$T0 timeout ${T:-3600} $PY $T0/perf/mol1/fair/run.py \
  --chips $C --part p150a --firmware "$FW" --rounds 3 --max-attempts 5 "$@"; }
cd $D/inputs
for arm in opendde:exact opendde:fast mol1:exact; do
  m=${arm%%:*}; mode=${arm#*:}; X=(); [ $mode = fast ] && X=(--fast)
  for c in 9TH6 9DBP 9LV4 9W3L; do
    o=$G/$m-$mode-$c; grep -qsx 0 $o.rc && continue
    run --model $m --arm $mode --input $D/inputs/$c.yaml --out $o --cli-extra --diffusion_samples 5 \
      --recycling_steps $([ $m = mol1 ] && echo 20 || echo 10) --msa_dir $D/msa/cache --msa_cache_only "${X[@]}" > $o.log 2>&1
    echo $? > $o.rc; say "$m $mode $c rc=$(cat $o.rc)"
    [ $(( $(end) - $(date +%s) )) -lt 600 ] && { say "window ending"; break 2; }
  done
done
cd /home/ttuser/spd/feed && python3 book.py rm $OWN
git -C $W worktree remove --force $T0
say done; echo ok > $G/DONE
