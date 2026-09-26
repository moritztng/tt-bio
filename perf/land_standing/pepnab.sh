#!/bin/bash
# The fused-hifi fold A/B on a REAL target, under the discipline the tiled-CDK2 number had.
#
# The ground-truth accuracy is settled: 0.055552 A of movement, 10.8x under the 0.60 A bar, and
# the lever's arm lands 0.004867 A CLOSER to the deposited 3B34. What is not settled is the speed
# on a real target -- the +8.471 s is a tiled-CDK2 number at 832 tokens -- and the seed floor on
# this fixture, which the 0.0556 A has to be read against.
#
# Six legs alternating so drift lands on both arms, AICLK sampled DURING each leg, board-pair
# sibling (card 2) watched on the same cadence. Then one extra OFF leg at seed 1: the seed floor
# is off-arm-to-off-arm, and without it 0.0556 A has a bar but no floor beside it.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/pepnab
MSA=$WT/perf/land_standing/out/narrowq_openbind_pepn/msa
IN=$WT/perf/land_standing/fixtures/pepn_3b34.yaml
CARD=3
SIB=2
mkdir -p "$OUT"
cd "$WT" || exit 1

leg () {
  n=$1
  v=$2
  seed=$3
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
      TT_BIO_TRIATT_FUSED_HIFI="$v" \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict --model openfold3 \
        --msa_dir "$MSA" --msa_cache_only --sampling_steps 20 --diffusion_samples 1 \
        --seed "$seed" --out_dir "$OUT/res_$n" "$IN" > "$OUT/log_$n.txt" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill $s 2>/dev/null
  med=$(sort -n "$OUT/clk_$n.txt" | awk '{a[NR]=$1} END{print a[int(NR/2)]}')
  sib=$(sort -n "$OUT/clk_$n.txt" | awk '{b[NR]=$2} END{print b[NR]}')
  ld=$(sort -n "$OUT/clk_$n.txt" | awk '{c[NR]=$3} END{print c[int(NR/2)]}')
  echo "LEG $n fuse=$v seed=$seed rc=$rc wall=$(echo "$t1 - $t0" | bc) aiclk=$med sib_max=$sib load=$ld"
}

# One discarded warm-up so the first timed leg is not the cold model load.
leg warm 0 0
for i in 1 2 3; do
  leg "off$i" 0 0
  leg "on$i" 1 0
done
leg offseed1 0 1
echo PEPNAB_DONE
