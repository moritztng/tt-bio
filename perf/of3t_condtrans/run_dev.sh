#!/usr/bin/env bash
# One diffusion-scope device arm per invocation, fresh process, one device context, card 0.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-condtrans
cd "$W" || exit 1
source /home/ttuser/tt-bio-dev/env/bin/activate
export OMP_NUM_THREADS=${OMP:-4}
O=/tmp/of3t/condtrans
mkdir -p "$O" "$W/perf/of3t_condtrans"
TAG=$1; shift
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-condtrans \
timeout 2400 python3 perf/of3t_diffusion/device_gradient.py --structs all \
  --dump-per-tensor --dump-grads "$O/dev_${TAG}.pt" \
  --out-dir "$W/perf/of3t_condtrans" --tag "_${TAG}" "$@"
echo "=== dev $TAG exit $? $(date -u +%FT%TZ) ==="
