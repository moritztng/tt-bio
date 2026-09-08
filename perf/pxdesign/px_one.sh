#!/bin/bash
# usage: px_one.sh <chip> <wt> <yaml> <label> <ndesigns> <nstep>
chip=$1 wt=$2 yaml=$3 label=$4 nd=$5 ns=$6
R=/home/cust-team/mthuening/ceilpxd/runs; out=$R/$label; rm -rf "$out"; mkdir -p "$out"
export TT_METAL_LOGGER_LEVEL=FATAL HF_HUB_CACHE=/home/cust-team/models
export TT_BIO_LEASE_HOLDER=worker:ceiling-pxdesign OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TT_VISIBLE_DEVICES=$chip TT_BIO_LEASE_CARDS=$chip
export TT_BIO_LEASE_TIMEOUT=${TT_BIO_LEASE_TIMEOUT:-900}
cd "$wt" || exit 2
t0=$(date +%s.%N)
timeout 1800 ./env/bin/python -u -m tt_bio.main design "$yaml" --model pxdesign \
  --cache /home/cust-team/.boltz --num_designs "$nd" --n_step "$ns" --seed 42 \
  --out_dir "$out/designs" > "$out/run.log" 2>&1
rc=$?; t1=$(date +%s.%N)
n=$(find "$out/designs" -name "*.cif" 2>/dev/null | wc -l)
fit=$(grep -o '"fit_rmsd": [0-9.]*' "$out"/designs/*.json 2>/dev/null | sed 's/.*: //' | cut -c1-6 | tr '\n' ',')
err=$(grep -oE "Not enough space to allocate [0-9,]+ B DRAM|largest free block: [0-9]+ B|device contention" "$out/run.log" | head -2 | tr '\n' ' ')
printf 'RESULT %s chip=%s rc=%s wall_s=%.1f cifs=%s fit=%s err=%s\n' "$label" "$chip" "$rc" "$(echo "$t1-$t0"|bc)" "$n" "${fit:-none}" "${err:-none}"
