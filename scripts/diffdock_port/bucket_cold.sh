#!/bin/bash
# dfd-bucket: what a complex the box has never seen costs on a card, before and after the row grid.
#
#   bucket_cold.sh <chip> <node> <arm> <budget_s> [complexes...]
#
# <arm> names a worktree under ~/dfd-bucket/<arm> (main or grid) and its own private JIT cache, so a
# warm shared cache cannot flatter the cold number. Per complex it records wall seconds, the kernel
# variants the run compiled (one directory per kernel name and compile-argument hash), the AICLK sampled during it and the box load.
# Output: ~/dfd-bucket/out/<arm>.log, <arm>.json, <arm>.aiclk.
set -u
chip=$1 node=$2 arm=$3 budget=$4; shift 4
C=${*:-"5S8I_2LY 6M2B_EZO 5SB2_1K2"}
R=$HOME/dfd-bucket; OUT=$R/out; L=$OUT/$arm.log; PB=$HOME/dfd-eval/pb/posebusters_benchmark_set
mkdir -p $OUT
exec 9>$OUT/$arm.runlock; flock -n 9 || exit 0
[ -e $HOME/fleet/HANDBACK ] && { echo "$arm: HANDBACK present, not run" >> $L; exit 0; }
n=$(sudo -n fuser /dev/tenstorrent/$node 2>/dev/null | wc -w)
[ "$n" = 0 ] || { echo "$arm: chip $chip (node $node) has $n opener(s), not run" >> $L; exit 0; }

cd $R/$arm || exit 1
CACHE=$OUT/cache-$arm; rm -rf $CACHE; mkdir -p $CACHE
CLK=/sys/class/tenstorrent/tenstorrent\!$node/tt_aiclk
( while :; do echo "$(date -u +%FT%TZ) $(cat $CLK 2>/dev/null) $(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done ) > $OUT/$arm.aiclk 2>/dev/null &
clk=$!
trap 'kill $clk 2>/dev/null' EXIT

export PYTHONPATH=$PWD DFD_CKPT=$HOME/dfd/data/ckpt/extracted TT_METAL_CACHE=$CACHE
export TT_VISIBLE_DEVICES=$chip TT_BIO_LEASE_CARDS=$chip TT_BIO_LEASE_HOLDER=worker:dfd-bucket
echo "$arm: start $(date -u +%FT%TZ) chip $chip node $node head $(git rev-parse --short HEAD) cache $CACHE $(uptime)" >> $L
echo "[" > $OUT/$arm.json
end=$(( $(date +%s) + budget ))
for name in $C; do
  [ $(date +%s) -lt $((end - 300)) ] || { echo "$arm: budget out before $name" >> $L; break; }
  before=$(find $CACHE -mindepth 5 -maxdepth 5 -type d -path '*/kernels/*' 2>/dev/null | wc -l)
  t0=$(date +%s.%N)
  timeout $((end - $(date +%s))) $HOME/fleet/env/bin/python -m tt_bio.main dock \
      $PB/$name/${name}_protein.pdb --ligand $PB/$name/${name}_ligand.sdf \
      --name $name --out_dir $OUT/poses-$arm --seed 0 >> $L 2>&1
  rc=$?
  t1=$(date +%s.%N)
  after=$(find $CACHE -mindepth 5 -maxdepth 5 -type d -path '*/kernels/*' 2>/dev/null | wc -l)
  by=$(find $CACHE -mindepth 5 -maxdepth 5 -type d -path '*/kernels/*' -newermt "@${t0%.*}" -printf '%h\n' 2>/dev/null \
       | sed 's|.*/||' | sort | uniq -c | sort -rn | head -8 | tr '\n' ';')
  clkd=$(awk -v a="${t0%.*}" '{print $2}' $OUT/$arm.aiclk | sort -n | uniq -c | tr '\n' ' ')
  printf '{"name":"%s","rc":%d,"wall_s":%.1f,"kernels_before":%d,"kernels_after":%d,"new_kernels":%d,"by_kernel":"%s","aiclk":"%s","load":"%s"},\n' \
      "$name" "$rc" "$(echo "$t1 - $t0" | bc)" "$before" "$after" "$((after-before))" "$by" "$clkd" "$(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT/$arm.json
  echo "$arm: $name rc=$rc wall=$(echo "$t1 - $t0" | bc)s new_kernels=$((after-before)) $(date -u +%FT%TZ)" >> $L
done
echo '{"end":"'"$(date -u +%FT%TZ)"'"}]' >> $OUT/$arm.json
echo "$arm: done $(date -u +%FT%TZ); aiclk during: $(cut -d' ' -f2 $OUT/$arm.aiclk | sort -n | uniq -c | tr '\n' ' ')" >> $L
