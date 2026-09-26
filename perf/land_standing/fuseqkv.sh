#!/bin/bash
# TT_BIO_TRIATT_FUSE_QKV: reach and firing first, then the fold.
#
# "ROOF Phase A: the qkv projection moved inside this kernel." The pair it replaces reads q, k and
# v where the fused form reads x, so at boltz-2's 512 aa triangle attention the pair moves
# 603.9 -> 402.7 MB. Off by default, held only as "a release-gated arm: it changes which
# arithmetic a shipped call reaches" -- a condition, not a refusal, and conditions are what this
# row exists to satisfy.
#
# Six legs alternating so drift lands on both arms, AICLK sampled DURING each leg, sibling card
# watched. FUSE_REJECTS is the firing proof: with the flag ON, a non-empty dict says the arm
# declined and names why, and an empty one with served calls says it ran.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/fuseqkv
IN=$WT/perf/land_standing/out/widekfire/inputs/aa384/cdk2apo_384.yaml
CARD=3
SIB=2
mkdir -p "$OUT"
cd "$WT" || exit 1

leg () {
  n=$1
  v=$2
  ( while true; do
      printf '%s %s %s\n' \
        "$(cat /sys/class/tenstorrent/tenstorrent!$CARD/tt_aiclk 2>/dev/null)" \
        "$(cat /sys/class/tenstorrent/tenstorrent!$SIB/tt_aiclk 2>/dev/null)" \
        "$(cut -d' ' -f1 /proc/loadavg)"
      sleep 1
    done > "$OUT/clk_$n.txt" ) &
  s=$!
  t0=$(date +%s.%N)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      PYTHONPATH="$WT/perf/land_standing/out/widekfire/hook" \
      TT_STOCK_STATS_DUMP="$OUT/stats_$n.jsonl" TT_BIO_TRIATT_FUSE_QKV="$v" \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict --model boltz2 \
        --single_sequence --sampling_steps 20 --diffusion_samples 1 \
        --out_dir "$OUT/res_$n" "$IN" > "$OUT/log_$n.txt" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill $s 2>/dev/null
  med=$(sort -n "$OUT/clk_$n.txt" | awk '{a[NR]=$1} END{print a[int(NR/2)]}')
  sib=$(sort -n "$OUT/clk_$n.txt" | awk '{b[NR]=$2} END{print b[NR]}')
  ld=$(sort -n "$OUT/clk_$n.txt" | awk '{c[NR]=$3} END{print c[int(NR/2)]}')
  cif=$(sha256sum "$OUT/res_$n"/boltz2_results_*/structures/*.cif 2>/dev/null | head -1 | cut -c1-16)
  echo "LEG $n fuse=$v rc=$rc wall=$(echo "$t1 - $t0" | bc) aiclk=$med sib=$sib load=$ld cif=$cif"
}

for i in 1 2 3; do
  leg "off$i" 0
  leg "on$i" 1
done
echo FUSE_DONE
