#!/usr/bin/env bash
# Item 2 of the token-bucket landing: record the p300c/qb2 perf rows for the three models the flip
# reaches, with the bucket at its shipped default (ON).
#
# The rows move because perf_regression's input is examples/trpcage.yaml at 20 aa, which the bucket
# pads 20 -> 64. That is the most extreme relative pad anywhere in the gate and it is a per-call
# cost, not padded-area scaling: pad multiple 32 recovers 2.6 of 16 points at 20 aa and would
# recover nothing at 298. At the sizes anyone folds for a result the same flip is free (512 aa,
# pad 0, byte-identical) or a win (298 aa, +4.8 % protenix-v2 / +5.97 % opendde). The GO decision
# and the GOALS.md SIZE GENERALITY reading behind it are in
# state/protenix-opendde-token-bucket-flip-measure.md.
#
# --update-baseline writes the DETECTED MACHINE's block (cards.p300c.machines.tt-quietbox2.models),
# so this does not touch the card-level rows every other p300c machine reads.
#
# The load guard is the point of having a script at all. The same leg read -16.0 % at loadavg 4.1
# and -61.4 % at loadavg 15-23 with a co-tenant on another card, so a write taken while the box is
# loud is worse than leaving the arm red for a day: it is that failure with a commit behind it.
set -u
: "${CARD:?set CARD to this launch grant}"
WT=$(cd "$(dirname "$0")/../.." && pwd)
SLUG=${SLUG:-$(basename "$WT")}
CEILING=${CEILING:-5.0}
# v0.7.0 raised the declared pins (transformers>=5.5.0); tt-bio-dev/env is on 4.57.6.
PY=${GATE_PYTHON:-/home/ttuser/.coworker/rel070/relvenv/bin/python3}
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
# without tt-smi on PATH the card detect falls back to sysfs and can report card_type unknown,
# which perf_regression treats as NO BASELINE.
export PATH=/home/ttuser/.local/bin:/home/ttuser/tt-bio/env/bin:$PATH
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:$SLUG"
OUT=perf/tokenbucket/rebaseline
mkdir -p $OUT
NOTE="token-bucket pad on by default: the trunk token axis is padded to a multiple of 64, masked, \
and sliced back on exit, because both the stock and the fused SDPA read padded key columns at a \
bias of zero. This input is trpcage at 20 aa, which the pad takes to 64, so it carries the most \
extreme relative pad in the gate as a fixed per-call cost. At 298 aa the same flip is +4.8 to \
+6.0 %, at 512 aa it is byte-identical. See state/protenix-opendde-token-bucket-flip-measure.md."

quiet() {
  for _ in $(seq 60); do
    l=$(cut -d' ' -f1 /proc/loadavg)
    awk -v a="$l" -v c="$CEILING" 'BEGIN{exit !(a+0<=c+0)}' && { echo "load $l ok"; return 0; }
    echo "$(date -Is) load $l > $CEILING, waiting"; sleep 30
  done
  return 1
}

for m in protenix-v2 opendde opendde-abag; do
  echo "=== $(date -Is) $m"
  quiet || { echo "HOLD $m: host never went quiet"; continue; }
  pre=$(cut -d' ' -f1-3 /proc/loadavg)
  env $LEASE PYTHONPATH=$WT ESM_ROOT=$ESM_ROOT "$PY" -u scripts/perf_regression.py \
      --model "$m" --update-baseline --note "$NOTE" > "$OUT/$m.log" 2>&1
  rc=$?
  echo "$(date -Is) $m rc=$rc load $pre -> $(cut -d' ' -f1-3 /proc/loadavg)"
  tail -4 "$OUT/$m.log"
done
echo "=== $(date -Is) ALL DONE"
git -C "$WT" diff --stat docs/perf_baselines.json
