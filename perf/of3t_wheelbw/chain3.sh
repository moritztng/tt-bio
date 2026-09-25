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
mkdir -p /tmp/of3t/of3t-wheelbw
cd "$WT" || exit 1
export PYTHONPATH="$WT:${PYTHONPATH:-}"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-wheelbw

# The status that matters is python's, and a pipeline hands back the LAST stage's -- so
# `python ... | grep -v DEBUG` reports the grep's success and a DeviceInUseError reads as a
# pass. Log to a file, keep python's own rc, then filter for the reader.
# A non-zero rc is not evidence of contention, and reading it as one is how this chain spent
# ~25 minutes holding pc's only Blackhole card re-running a TypeError at six minutes a cycle.
# Only DeviceInUseError means "come back later"; everything else is this row's own bug and
# retrying it cannot succeed. `fuser` is no help either -- it printed an empty holder list on
# every one of those cycles -- so the holder comes out of the lease's own message.
step() {
  local label=$1; shift
  local log
  for try in $(seq 1 40); do
    log=/tmp/of3t/of3t-wheelbw/$label.$try.log
    echo "=== $label try $try $(date -u +%FT%TZ) ==="
    timeout 2400 "$@" > "$log" 2>&1
    local rc=$?
    grep -v 'DEBUG *|\|Config{' "$log" | tail -40
    if [ $rc -eq 0 ]; then
      echo "=== $label OK rc=0 $(date -u +%FT%TZ) ==="; return 0
    fi
    if ! grep -q 'DeviceInUseError' "$log"; then
      echo "=== $label FAULT rc=$rc, not contention -- no retry $(date -u +%FT%TZ) ==="
      return "$rc"
    fi
    echo "=== $label busy: $(grep -o 'in use by [^;]*' "$log" | tail -1) ==="
    sleep 60
  done
  echo "=== $label GAVE UP after 40 tries ==="
  return 1
}

# The two VJP arms are BANKED -- `out/vjp_float32_r2.json` and `out/vjp_bfloat16_r2.json`,
# fd_pass 10/10 at both dtypes, pc card 0 p150a. Re-running them would spend the fleet's only
# Blackhole card on a question already answered, so what is left is the census and the two
# speed arms. To retake them, uncomment:
#   step vjp_f32  $PY perf/of3t_wheelbw/vjp.py --dtype float32  --out "$OUT/vjp_float32_r2.json"
#   step vjp_bf16 $PY perf/of3t_wheelbw/vjp.py --dtype bfloat16 --out "$OUT/vjp_bfloat16_r2.json"
step census    $PY perf/of3t_wheelbw/census.py --tokens 384 --out "$OUT/census_384.json"
step speed_f32 $PY perf/of3t_wheelbw/speed.py --card 0 --dtype float32 \
       --shape 1,384,384,128 --reps 5 --iters 20 --out "$OUT/speed_f32_pair384.json"
step speed_bf16 $PY perf/of3t_wheelbw/speed.py --card 0 --dtype bfloat16 \
       --shape 1,384,384,128 --reps 5 --iters 20 --out "$OUT/speed_bf16_pair384.json"
echo "CHAIN3 DONE $(date -u +%FT%TZ)"
