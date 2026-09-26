#!/bin/bash
# Does main ALREADY ship, at a neighbouring length, the route difference the dividing-k lever
# introduces at 832?
#
# 832 is the only length OpenFold3's trunk can present where the fused HiFi route declines. Its
# neighbours 704, 1088, 1216 and 1472 all serve it by default today. So the fused-vs-materialised
# divergence is something main already ships everywhere EXCEPT 832, and the size of it at 704 is
# the yardstick 832's 2.152e-01 / 8.724e-02 has been missing.
#
# Three legs: shipped default, the same length forced onto the materialised fall-back, shipped
# default again. The third leg is the control and must reproduce the first exactly -- the 832 work
# established the trunk is bit-deterministic across runs, and this re-establishes it here rather
# than assuming it transfers.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/neighbour704
IN=$OUT/inputs/aa704/cdk2apo_704.yaml
HOOK=$OUT/hook
CARD=3
cd "$WT" || exit 1

run_leg () {
  tag=$1
  ab=$2
  dump=$OUT/trunk_$tag.jsonl
  rm -f "$dump"
  ( while true; do
      printf '%s %s\n' "$(cat /sys/class/tenstorrent/tenstorrent!$CARD/tt_aiclk 2>/dev/null)" \
                       "$(cut -d' ' -f1 /proc/loadavg)"
      sleep 1
    done > "$OUT/clk_$tag.txt" ) &
  sampler=$!
  t0=$(date +%s.%N)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      PYTHONPATH="$HOOK" TT_TRUNK_DUMP="$dump" TT_BIO_TRIATT_SDPA_HIFI_AB="$ab" \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict --model openfold3 --single_sequence \
        --sampling_steps 20 --diffusion_samples 1 --out_dir "$OUT/res_$tag" "$IN" \
      > "$OUT/log_$tag.txt" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill $sampler 2>/dev/null
  echo "LEG $tag ab='$ab' rc=$rc wall=$(echo "$t1 - $t0" | bc)s dump_lines=$(wc -l < "$dump" 2>/dev/null || echo 0)"
}

run_leg fusedA ""
run_leg matB   "-openfold3.trunk"
run_leg fusedC ""
echo ALL_LEGS_DONE
