#!/usr/bin/env bash
# of3t-cotterm: score each counterfactual on the trunk, with of3t-apbleaf's OWN cf.py.
# Same scorer that produced CF_perfect = 0.6822397912, so the numbers are comparable rather
# than merely similar. Frame: ours against upstream 0.4.3's own bf16 autocast at padded 384.
#   runcf.sh <DEVTAG> <CH> [<CH> ...]
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-cotterm
O=/tmp/of3t/of3t-cotterm
D=/home/ttuser/of3t_frame384
cd "$W"
TAG=$1; shift
for C in "$@"; do
  echo "=== cf $TAG $C start $(date -u +%FT%TZ) host=$(hostname) ==="
  OMP_NUM_THREADS=16 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_apbleaf/cf.py \
    --f64 "$D/ref_f64_n384.pt" --bf16auto "$D/ref_bf16auto_n384.pt" \
    --ours "$O/dev_${TAG}_n384.pt" --mech-cf "$O/CF_${TAG}_N384_${C}.pt" \
    --out "$W/perf/of3t_cotterm/CFSCORE_${TAG}_${C}.json"
  echo "=== cf $C exit $? $(date -u +%FT%TZ) ==="
done
