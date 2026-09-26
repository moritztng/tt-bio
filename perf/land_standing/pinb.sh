#!/bin/bash
# Re-run the pinned-pair leg only. The first attempt keyed its patch on the imported module name
# and never fired: `tenstorrent.py` does `from . import triatt_sdpa`, so that substring never
# reaches __import__. The hook now asks sys.modules. hifi_picks is the firing proof -- a leg that
# still reports (416, 416) did not take the pin, whatever TT_FORCE_FUSED_PAIR says.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/singlek832
IN=$WT/perf/land_standing/out/plddt832/inputs/aa832/cdk2apo_832.yaml
CARD=3
cd "$WT" || exit 1

run_leg () {
  tag=$1
  pair=$2
  dump=$OUT/stats_$tag.jsonl
  rm -f "$dump"
  ( while true; do
      printf '%s %s\n' "$(cat /sys/class/tenstorrent/tenstorrent!$CARD/tt_aiclk 2>/dev/null)" \
                       "$(cut -d' ' -f1 /proc/loadavg)"
      sleep 1
    done > "$OUT/clk_$tag.txt" ) &
  sampler=$!
  t0=$(date +%s.%N)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      PYTHONPATH="$OUT/hook" TT_STATS_DUMP="$dump" TT_FORCE_FUSED_PAIR="$pair" \
      TT_BIO_TRIATT_MASK_Q_SPLIT_MAX=768 \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict --model openfold3 \
        --single_sequence --sampling_steps 20 --diffusion_samples 1 \
        --out_dir "$OUT/res_$tag" "$IN" > "$OUT/log_$tag.txt" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill $sampler 2>/dev/null
  echo "LEG $tag pair='${pair:-kernel order}' rc=$rc wall=$(echo "$t1 - $t0" | bc)s"
  grep -o '"plddt": [0-9.]*\|"ptm": [0-9.]*' \
    "$OUT/res_$tag/openfold3_results_cdk2apo_832/results.json" 2>/dev/null | tr '\n' ' '
  echo
  tail -1 "$dump" 2>/dev/null
}

run_leg pinB2 "64,832"
echo ALL_LEGS_DONE
