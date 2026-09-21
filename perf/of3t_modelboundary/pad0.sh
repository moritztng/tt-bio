#!/usr/bin/env bash
# of3t-modelboundary: the same n384 RENORM arm with the boundary's PAD rows and columns zeroed.
# The whole point: at n384 the model's batch is 56 real tokens of 384, so 97.9 % of the pair
# cells are pad. If our trunk's gradient excess (r = 2.10 against the model's float64 reference)
# comes from the pad region, zeroing the pad input collapses it. If it does not, the pad is not
# the mechanism and the disagreement is on the 56 real tokens.
set -uo pipefail
cd /home/ttuser/.coworker/wt/of3t-modelboundary
O=/tmp/of3t/of3t-modelboundary
CLK=$O/aiclk_PAD0.txt
: > "$CLK"
( while true; do
    /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
      | python3 -c 'import sys,json;print(json.load(sys.stdin)["device_info"][0]["telemetry"]["aiclk"].strip())' \
      >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
S=$!
source /home/ttuser/tt-bio-dev/env/bin/activate
date -u +%FT%TZ
TT_BIO_SOFTMAX_BW_RENORM=1 TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
TT_BIO_LEASE_HOLDER=worker:of3t-modelboundary OMP_NUM_THREADS=8 \
timeout 3000 python3 perf/of3t_bwdaccum/dev_cot.py --lever none \
  --boundary /home/ttuser/of3t_trunk043ref/boundary_n384.pt \
  --cap-last /home/ttuser/of3t_gradients/cap/block47_boundary.pt \
  --out "$O/dev_RENORM_n384_pad0.pt" \
  --report perf/of3t_modelboundary/DEV_RENORM_n384_pad0.json \
  --arm flipped --crop 0 --pad-scale 0
echo "EXIT=$?"
kill "$S" 2>/dev/null
date -u +%FT%TZ
printf 'AICLK during: '
sort -n "$CLK" | awk '{a[NR]=$1} END{printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]}'
echo PAD0_DONE
