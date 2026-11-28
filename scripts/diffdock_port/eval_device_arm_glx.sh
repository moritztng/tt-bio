#!/bin/bash
# dfd-eval phase 2 on one anx-glx chip: the same forms, ids, seed and recipe as eval_device_arm.sh,
# without qb2's booking calendar. The chip is held in state/anx-glx/HOLDER instead, and the arm
# stops by itself at <stop_epoch> (a whole-box window) or when ~/fleet/HANDBACK appears.
#
#   eval_device_arm_glx.sh <name> <forms, e.g. gemm,tile> <chip> <stop_epoch> <ids.json>
#
# Chips 16-31 open the device node 8 away from their id (state/anx-glx/USE.md), so the idle check
# and the AICLK read go to the node, not the id. The dock is resumable, so a stop costs only the
# complex in flight.
set -uo pipefail
name=${1:?name} etps=${2:?forms} chip=${3:?chip} stop=${4:?stop epoch} ids=${5:?ids.json}
R=$HOME/dfd-eval; WT=$R/wt; OUT=$R/device; PY=$HOME/fleet/env/bin/python
node=$chip; [ $chip -ge 24 ] && node=$((chip - 8)); [ $chip -ge 16 ] && [ $chip -le 23 ] && node=$((chip + 8))
log=$OUT/$name.log
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $name: $*" >> "$log"; }
[ -e "$OUT/$name.done" ] && exit 0
exec 9>"$OUT/$name.armlock"; flock -n 9 || exit 0
[ -e $HOME/fleet/HANDBACK ] && { say "HANDBACK present, not run"; exit 0; }
[ -s /proc/driver/tenstorrent/$node/pids ] && { say "chip $chip (node $node) has an opener, not run"; exit 0; }
cd "$WT" || exit 1
export PYTHONPATH=$WT DFD_PB=$R/pb/posebusters_benchmark_set
export TT_VISIBLE_DEVICES=$chip TT_BIO_LEASE_CARDS=$chip TT_BIO_LEASE_HOLDER=worker:dfd-eval
( while sleep 20; do [ -e $HOME/fleet/HANDBACK ] && pkill -INT -P $$ -f pose_eval; done ) &
watch=$!
forms=(${etps//,/ })
for idx in "${!forms[@]}"; do
  etp=${forms[$idx]}; poses=$OUT/poses-$etp
  [ -e "$poses/DONE" ] && continue
  left=$(( stop - $(date +%s) ))
  [ $left -le 120 ] && { say "$etp: no time left, resumes next window"; break; }
  share=$(( left / (${#forms[@]} - idx) ))
  say "start $etp chip $chip node $node budget ${share}s $(uptime)"
  s=$(date +%s)
  timeout -s INT -k 60 $share $PY scripts/diffdock_port/pose_eval.py dock --ids "$ids" --out "$poses" \
    --device tt --edge-tp "$etp" --seed 42 \
    --clk-sysfs "/sys/class/tenstorrent/tenstorrent!$node/tt_aiclk" > "$OUT/$name-$etp.out" 2>&1
  rc=$?
  cat "$OUT/$name-$etp.out" >> "$log"
  grep -q '^DOCK_DONE' "$OUT/$name-$etp.out" && touch "$poses/DONE"
  say "$etp rc=$rc wall_s=$(( $(date +%s) - s )) rank1=$(ls "$poses"/*/rank1.sdf 2>/dev/null | wc -l) $(uptime)"
done
kill $watch 2>/dev/null
alldone=1; for e in "${forms[@]}"; do [ -e "$OUT/poses-$e/DONE" ] || alldone=0; done
[ $alldone = 1 ] && { touch "$OUT/$name.done"; say "all forms done"; }
say "released"
