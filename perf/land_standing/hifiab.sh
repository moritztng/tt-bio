#!/bin/bash
# Fold A/B for TT_BIO_TRIATT_FUSED_HIFI at 832 tokens on OpenFold3, where it takes served calls
# from 384 to 440 -- the 56 sites that pass fused_hifi=None and follow the process-wide flag.
# Six legs alternating off/on so drift lands on both arms; the three OFF legs give the A/A floor
# that the effect has to clear. AICLK sampled every second DURING each leg from the class node,
# with the board-pair sibling (card 2) watched on the same cadence, because that pair shares a
# power budget and has destroyed a reading before.
#
# 1.1554x is the op-level number this lever already carries. Op-level ratios do not transfer to
# the fold, which is the whole reason for this script.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/hifiab
IN=$WT/perf/land_standing/out/deepfix/cdk2_deep_832.yaml
CARD=3
SIB=2
mkdir -p "$OUT"
cd "$WT" || exit 1

leg () {
  n=$1
  up=$2
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
      TT_STOCK_STATS_DUMP="$OUT/stats_$n.jsonl" TT_BIO_TRIATT_FUSED_HIFI="$up" \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict --model openfold3 \
        --msa_dir /home/ttuser/.coworker/wt/land-standing/perf/land_standing/out/deepfix/msa --msa_cache_only --sampling_steps 20 --diffusion_samples 1 \
        --out_dir "$OUT/res_$n" "$IN" > "$OUT/log_$n.txt" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill $s 2>/dev/null
  med=$(sort -n "$OUT/clk_$n.txt" | awk '{a[NR]=$1} END{print a[int(NR/2)]}')
  sib=$(sort -n "$OUT/clk_$n.txt" | awk '{b[NR]=$2} END{print b[NR]}')
  ld=$(sort -n "$OUT/clk_$n.txt" | awk '{c[NR]=$3} END{print c[int(NR/2)]}')
  cif=$(sha256sum "$OUT/res_$n"/openfold3_results_*/structures/*.cif 2>/dev/null | head -1 | cut -c1-16)
  echo "LEG $n up=$up rc=$rc wall=$(echo "$t1 - $t0" | bc) aiclk_med=$med sib_max=$sib load_med=$ld cif=$cif"
}

for i in 1 2 3; do
  leg "off$i" 0
  leg "on$i" 1
done
echo AB_DONE
