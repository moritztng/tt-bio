#!/bin/bash
# Device proof v2: run each tree FROM its own dir (python -m puts cwd first on sys.path; v1 ran the branch twice).
# --write_pae so full_data_sample_<k>.json (token_pair_pae) lands. popvax branch run twice = same-code noise control.
set -u
WT=/home/ttuser/.coworker/wt/jfc-opendde
P=$WT/_proof
rm -f "$P/ALL_DONE3"
run() {  # inp code tag [dram]
  local inp=$1 code=$2 tag=$3 dram=${4:-}
  local src=$WT; [ $code = base ] && src=$P/base
  local out=$P/v3_${inp}_${tag}; rm -rf "$out"; mkdir -p "$out"
  ( while :; do echo "$(date +%s) $(cat /sys/class/tenstorrent/tenstorrent!0/tt_aiclk)"; sleep 1; done ) > "$out/aiclk.log" &
  local sp=$!
  local t0=$(date +%s.%N)
  ( cd "$src" && env ${dram:+TT_BIO_DRAM_PEAK=$out/dram.txt} TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
      TT_BIO_LEASE_HOLDER=worker:jfc-opendde timeout 2400 /usr/bin/time -v ~/tt-bio-dev/env/bin/python -m tt_bio.main \
      predict "$P/$inp.yaml" --model opendde --single_sequence --diffusion_samples 5 --seed 101 --output_format cif \
      --write_pae --out_dir "$out/res" ) > "$out/run.log" 2>&1
  echo "rc=$? wall=$(echo "$(date +%s.%N) - $t0" | bc)" > "$out/rc"
  kill $sp
}

run popvax base base
run popvax branch branch
run popvax base base2
run popvax branch branch2

touch "$P/ALL_DONE3"
