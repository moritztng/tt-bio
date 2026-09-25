#!/bin/bash
# J4's device leg: dtype probe -> VJP (fp32, bf16) -> speed, on pc's one Blackhole.
# pc carries ONE card, UMD chip 0, p150a. The brief's "physical 5" does not exist here.
#
# It does NOT gate on `fuser` first. pc's single card is contested by four of3t rows and it
# changes hands in seconds -- measured this pass, three different holders inside four minutes,
# each one acquiring inside a 10 s poll of the last -- so a poll-then-launch gate loses that
# race every time. This launches and lets the device open queue instead, and records who else
# held the node around each step, so a timing taken beside another row's arm shows up as one
# rather than being quoted as one.
set -u
WT=/home/moritz/.coworker/wt/of3t-wheelbw
OUT=$WT/perf/of3t_wheelbw/out
PY=/home/moritz/tt-bio/env/bin/python3
mkdir -p "$OUT" /tmp/of3t/of3t-wheelbw
cd "$WT" || exit 1
export PYTHONPATH="$WT:${PYTHONPATH:-}"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-wheelbw

holders() { fuser /dev/tenstorrent/0 2>&1 | tr '\n' ' '; }

run() {
  echo "=== $* ==="
  echo "holders before: $(holders)  $(date -u +%FT%TZ)"
  timeout 2400 "$@" 2>&1 | grep -v 'DEBUG *|\|Config{'
  echo "holders after:  $(holders)  $(date -u +%FT%TZ)"
}

run $PY perf/of3t_wheelbw/probe_dtypes.py "$OUT/dtypes.json"
run $PY perf/of3t_wheelbw/vjp.py --dtype float32  --out "$OUT/vjp_f32.json"
run $PY perf/of3t_wheelbw/vjp.py --dtype bfloat16 --out "$OUT/vjp_bf16.json"
run $PY perf/of3t_wheelbw/census.py --tokens 384 --out "$OUT/census_384.json"
run $PY perf/of3t_wheelbw/speed.py --card 0 --dtype float32 --shape 1,384,384,128 \
        --reps 5 --iters 20 --out "$OUT/speed_f32_pair384.json"
run $PY perf/of3t_wheelbw/speed.py --card 0 --dtype bfloat16 --shape 1,384,384,128 \
        --reps 5 --iters 20 --out "$OUT/speed_bf16_pair384.json"
echo "CHAIN DONE $(date -u +%FT%TZ)"
