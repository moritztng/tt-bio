#!/bin/bash
# J4's device leg: wait for the card, then dtype probe -> VJP (fp32, bf16) -> speed.
# pc carries ONE Blackhole, UMD chip 0, p150a. The brief's "physical 5" does not exist here.
set -u
WT=/home/moritz/.coworker/wt/of3t-wheelbw
OUT=$WT/perf/of3t_wheelbw/out
PY=/home/moritz/tt-bio/env/bin/python3
mkdir -p "$OUT" /tmp/of3t/of3t-wheelbw
cd "$WT" || exit 1
export PYTHONPATH="$WT:${PYTHONPATH:-}"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-wheelbw

# The card is shared with of3t-tapedfwd's fires.py this pass. Two processes on one Blackhole
# is how this fleet wedges a host, and a timing taken beside another row's arm is not a
# timing, so wait rather than contend.
for _ in $(seq 1 360); do
  if ! fuser /dev/tenstorrent/0 >/dev/null 2>&1; then break; fi
  sleep 10
done
if fuser /dev/tenstorrent/0 >/dev/null 2>&1; then
  echo "CARD STILL HELD after 60 min: $(fuser -v /dev/tenstorrent/0 2>&1 | tail -2)"
  exit 3
fi
echo "card free at $(date -u +%FT%TZ)"

run() { echo "=== $* ==="; timeout 1800 "$@" 2>&1 | grep -v "DEBUG *|\|Config{"; }

run $PY perf/of3t_wheelbw/probe_dtypes.py "$OUT/dtypes.json"
run $PY perf/of3t_wheelbw/vjp.py --dtype float32  --out "$OUT/vjp_f32.json"
run $PY perf/of3t_wheelbw/vjp.py --dtype bfloat16 --out "$OUT/vjp_bf16.json"
run $PY perf/of3t_wheelbw/speed.py --card 0 --dtype float32 --shape 1,384,384,128 \
        --reps 5 --iters 20 --out "$OUT/speed_f32_pair384.json"
run $PY perf/of3t_wheelbw/speed.py --card 0 --dtype bfloat16 --shape 1,384,384,128 \
        --reps 5 --iters 20 --out "$OUT/speed_bf16_pair384.json"
echo "CHAIN DONE $(date -u +%FT%TZ)"
