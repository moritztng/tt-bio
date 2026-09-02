#!/bin/bash
# Single-chip ladder runner. Hard-pinned to /dev/tenstorrent/5, which on this box is UMD 29 --
# TT_VISIBLE_DEVICES is a UMD id and UMD 5 is /dev/tenstorrent/21, a production-pool chip.
# One rung at a time, sequential, never a second chip.
set -u
H=/home/cust-team/mthuening
B=$H/ceilpx2
PY=$H/tt-bio/env/bin/python3.10
MSA=$H/abag_xm/msa_cache
NODE=5
UMD=29
JOBS="${1:?usage: one5.sh \"fix:1120s fix:1152s\"}"
mkdir -p "$B/claims5" "$B/out5"

node5_busy() { sudo lsof /dev/tenstorrent/$NODE 2>/dev/null | tail -n +2 | awk '{print $2}' | sort -u | head -1; }

for j in $JOBS; do
  tree=${j%%:*}; rung=${j##*:}; tag="${tree}_${rung}"
  [ -e "$B/claims5/$tag.done" ] && continue
  # Wait for my own chip to come free. Never take another.
  while [ -n "$(node5_busy)" ]; do sleep 30; done
  sleep 10
  [ -n "$(node5_busy)" ] && { echo "$(date -u +%FT%TZ) $tag: node $NODE retaken, retrying" >> "$B/one5.log"; sleep 60; }
  echo "$(date -u +%FT%TZ) start $tag on node $NODE (umd $UMD)" >> "$B/one5.log"
  SRC=$B/${tree}_src
  ( cd "$SRC"; s=$(date +%s)
    TT_VISIBLE_DEVICES=$UMD PYTHONPATH=$SRC OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      HF_HUB_CACHE=$H/models \
      timeout -k 30 7200 $PY -u -m tt_bio.main predict $B/inputs/px$rung.yaml \
      --model protenix-v2 --out_dir $B/out5/$tag --override --fast \
      --diffusion_samples 1 --max_parallel_samples 1 --seed 42 --host_threads 2 \
      --msa_dir $MSA --msa_cache_only > $B/$tag.log 2>&1
    rc=$?; secs=$(( $(date +%s) - s ))
    cifs=$(ls $B/out5/$tag/*results_*/structures/*.cif 2>/dev/null | wc -l)
    md5=$(cat $B/out5/$tag/*results_*/structures/*.cif 2>/dev/null | md5sum | cut -d' ' -f1)
    ask=$(grep -o "allocate [0-9]* B DRAM" $B/$tag.log 2>/dev/null | head -1 | tr -dc '0-9')
    printf '{"tree":"%s","rung":"%s","node":%s,"rc":%s,"secs":%s,"cifs":%s,"ask":%s,"md5":"%s"}\n' \
      "$tree" "$rung" "$NODE" "$rc" "$secs" "${cifs:-0}" "${ask:-0}" "$md5" >> "$B/results5.jsonl" )
  touch "$B/claims5/$tag.done"
done
echo "$(date -u +%FT%TZ) one5 ladder complete" >> "$B/one5.log"
