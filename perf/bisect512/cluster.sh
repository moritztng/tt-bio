#!/usr/bin/env bash
# Bracket the 08-21/08-22 trunk-change cluster, then attribute per commit inside it.
# A pre-08-27 commit can deadlock OpenDDE at 512 aa on pc's 13x10 grid; the 900s guard in
# fold_at.sh bounds that, and a wedged card is reset before the next arm so one hang cannot
# poison every later fold (which is exactly what happened on the first attempt).
WT=/home/moritz/.coworker/wt/opendde-512aa-numerics-drift-bisect
cd "$WT" || exit 1
run(){
  REPEAT=1 ./.bisect-out/fold_at.sh "$1" "$2" >> ".bisect-out/$1.log" 2>&1
  if [ ! -f ".bisect-out/$1.json" ]; then
    echo "=== $1 produced no json; resetting card 0 before next arm ===" >> .bisect-out/cluster.out
    timeout 180 ~/.local/bin/tt-smi -r 0 >> .bisect-out/cluster.out 2>&1
    sleep 20
  fi
  sleep 5
}
run BR_pre_78ed5a1e   78ed5a1e^
run BR_post_7f00b025  7f00b025
run C_78ed5a1e        78ed5a1e
run C_4e3d922a        4e3d922a
run C_f45efe85        f45efe85
git checkout -q wk/opendde-512aa-numerics-drift-bisect
echo "=== CLUSTER DONE $(date -u +%FT%TZ) ==="
