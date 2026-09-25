#!/bin/bash
# J4's remaining device leg, through tt-bio's device lease instead of a raw ttnn.open_device.
#
# The first chain opened the card with `ttnn.open_device` and so bypassed the lease entirely:
# it queued at the fd level beside of3t-tapedfwd's and of3t-bwattrib's arms rather than being
# refused, which is fine for an accuracy reading and is not a timing. `get_device` takes the
# lease and raises DeviceInUseError naming the holder, so each step below retries until the
# card is genuinely this row's.
set -u
WT=/home/moritz/.coworker/wt/of3t-wheelbw
OUT=$WT/perf/of3t_wheelbw/out
PY=/home/moritz/tt-bio/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH="$WT:${PYTHONPATH:-}"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-wheelbw

step() {
  local label=$1; shift
  for try in $(seq 1 40); do
    echo "=== $label try $try $(date -u +%FT%TZ) ==="
    if timeout 2400 "$@" 2>&1 | grep -v 'DEBUG *|\|Config{'; then
      echo "=== $label OK $(date -u +%FT%TZ) ==="; return 0
    fi
    echo "=== $label retry: $(fuser /dev/tenstorrent/0 2>&1) ==="
    sleep 60
  done
  echo "=== $label GAVE UP after 40 tries ==="
}

step vjp_f32   $PY perf/of3t_wheelbw/vjp.py --dtype float32  --out "$OUT/vjp_float32_r2.json"
step vjp_bf16  $PY perf/of3t_wheelbw/vjp.py --dtype bfloat16 --out "$OUT/vjp_bfloat16_r2.json"
step census    $PY perf/of3t_wheelbw/census.py --tokens 384 --out "$OUT/census_384.json"
step speed_f32 $PY perf/of3t_wheelbw/speed.py --card 0 --dtype float32 \
       --shape 1,384,384,128 --reps 5 --iters 20 --out "$OUT/speed_f32_pair384.json"
step speed_bf16 $PY perf/of3t_wheelbw/speed.py --card 0 --dtype bfloat16 \
       --shape 1,384,384,128 --reps 5 --iters 20 --out "$OUT/speed_bf16_pair384.json"
echo "CHAIN3 DONE $(date -u +%FT%TZ)"
