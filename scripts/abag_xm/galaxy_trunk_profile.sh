#!/bin/bash
# Profile the opendde-abag trunk's device DRAM at the campaign's real config: FULL depth,
# UNCAPPED, UNPAIRED. Reads the per-stage [DRAM] trace that tt_bio.tenstorrent.dram_peak
# writes, so the peak can be decomposed into floor + k*m_feat + pair_copies*z.
#
# --msa_cache_only is what makes the depth reproducible: --msa_dir alone is NOT a source, so
# the run would fall through to the online-server fallback, which also switches on the
# multi-chain paired search (both gated on want_msa) and lands 35-45% deeper than intended.
# With the flag nothing is searched and an uncached chain is an error, not a silent
# single-sequence fold. Unpaired also matches the shipped Blackhole benchmark, which is
# unpaired on every row -- pairing here would make the two datasets non-comparable.
#
# Usage: galaxy_trunk_profile.sh <target> <chip> [extra predict args...]
set -u
T=$1; CHIP=$2; shift 2
SRC=${TRUNK_PROFILE_SRC:-$HOME/mthuening/parity-src}
OUT=${TRUNK_PROFILE_OUT:-$HOME/mthuening/dram}
MSA=${TRUNK_PROFILE_MSA:-$HOME/mthuening/abag_xm/msa_cache}

mkdir -p "$OUT"
cd "$SRC" || exit 1
export PYTHONPATH=$SRC
export TT_BIO_DRAM_PEAK=$OUT/$T.dram.txt
# 64 cores shared by several concurrent profiles; omitting a cap has collapsed throughput
# three times in this campaign (memory: threadcap-blind-to-sibling-processes).
export OMP_NUM_THREADS=${PROFILE_THREADS:-8} MKL_NUM_THREADS=${PROFILE_THREADS:-8}
rm -f "$TT_BIO_DRAM_PEAK"

TT_VISIBLE_DEVICES=$CHIP /usr/bin/python3.10 -u -m tt_bio.main predict \
  "examples/abag_xm/$T.yaml" --model opendde-abag \
  --out_dir "$OUT/$T" --override \
  --diffusion_samples 1 --max_parallel_samples 1 --seed 42 \
  --host_threads "${PROFILE_THREADS:-8}" \
  --msa_dir "$MSA" --msa_cache_only "$@"
echo "EXIT=$? target=$T chip=$CHIP"
