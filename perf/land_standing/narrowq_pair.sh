#!/usr/bin/env bash
# Take ONE order-controlled interleaved pair of the narrow-q rf3 896 aa cell and bank it.
#
# WHY A PAIR AND NOT THE CELL. The four-rep cell wants ~19 minutes under the benchlock ceiling.
# On 2026-09-22 this row got 6: pre-flight passed at loadavg 0.13 at 22:05:45Z and of3t started
# stacking ref_grad.py, c64_score.py and model_scope.py at 22:11:40Z. Two of eight legs survived
# the ceiling cut, one complete rep, so the cell had no A/A floor and was refused -- after having
# spent the window. A pair is warm-up + two folds, about 5.5 minutes, which is a window this box
# actually hands out. Pairs accumulate in the bank and narrowq_bank.py assembles them.
#
# WHY THIS IS NOT A LOOSER TEST. Each pair is two ADJACENT folds in one process on one device
# open, so within-pair drift is cancelled exactly as it is inside a single-session cell. Pooling
# pairs taken in different windows can only ADD between-window variation to the A/A floor, so the
# assembled cell is scored against a floor at least as wide as a single-session one. The bar goes
# up, not down.
#
# ORDER IS THE THING TO GET RIGHT. fold_ab_flip alternates arm order by rep, and its own comment
# records why: the 768 aa negative control, a length where the policy provably cannot reach the
# fallback, read +2.542 % at 3.80x its A/A floor purely because the lever arm ran first in both
# pairs. Every banked pair is rep 1, so without --first-arm every pair would be off-first and
# that exact confound comes back. Pass alternating values; narrowq_bank.py REFUSES a bank that
# carries only one order.
#
# Usage: NQ_RUNG=<aa> narrowq_pair.sh off|on   (NQ_RUNG defaults to 896)
#
# THE RUNG IS AN ARGUMENT BECAUSE 896 IS NOT THE WHOLE LEVER. The policy fires at 30 of the
# 37 tile-aligned lengths from 256 to 1408 -- every one that is not a multiple of 256 -- and
# a default flipped on one length is the standing one-size defect. 1088 aa is the rung that
# can falsify it: it fires and L1 refuses MORE configs there, so it stresses the term the
# lever leans on hardest, and a narrower q_chunk re-reads K and V once more per chunk.
# 768 aa cannot inform the question either way -- the policy returns the identical tuple
# with the flag set or clear at every multiple of 256, so it is a no-op by construction.
set -u
FIRST="${1:-}"
case "$FIRST" in off|on) ;; *) echo "usage: $0 off|on"; exit 2;; esac

WT=/home/ttuser/.coworker/wt/land-standing
BANK=$WT/perf/land_standing/out/narrowq_bank
PY=/home/ttuser/tt-bio-dev/env/bin/python3
# CARD/SIB are overridable because the board pair, not the card, is what has to be idle: on
# 2026-09-22 22:34Z card 0 was free while its pair sibling card 1 ran of3t's dev_cot.py, and the
# pair guard refuses that correctly. Cards 2 and 3 were both idle. Both are p300c on one host, so
# a pair taken on either board is the same cell; pooling across boards can only widen the A/A
# floor, the same way pooling across windows does.
CARD=${NQ_CARD:-0}
SIB=${NQ_SIB:-1}
RUNG=${NQ_RUNG:-896}
cd "$WT" || exit 1
mkdir -p "$BANK"

"$PY" perf/c12_orchestrator/pair_guard/host_quiet.py || {
  echo "PRE-FLIGHT REFUSED: host not quiet."; exit 3; }
"$PY" perf/c12_orchestrator/pair_guard/pair_idle.py --card $CARD || {
  echo "PRE-FLIGHT REFUSED: board-pair sibling $SIB busy."; exit 3; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
echo "PRE-FLIGHT OK $STAMP rung $RUNG loadavg $(cut -d' ' -f1-3 /proc/loadavg) first-arm $FIRST"

export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing
ART=$BANK/pair_${STAMP}_r${RUNG}_${FIRST}_c${CARD}.json
JL=$BANK/pair_${STAMP}_r${RUNG}_${FIRST}_c${CARD}_contention.jsonl
: > "$JL"
"$PY" perf/pvx_gate_land/sample_contention.py --out "$JL" --interval 1.0 &
SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT

exec 9>/home/ttuser/.coworker/state/benchlock.flock
flock -n 9 || { echo "benchlock held by another row; refusing"; exit 3; }

"$PY" perf/xmsoftmax/fold_ab_flip.py --models rf3 --rungs "$RUNG" --reps 1 \
  --flag TT_BIO_TRIATT_NARROW_Q_FALLBACK --off-value 1 --fold-timeout-s 600 \
  --first-arm "$FIRST" \
  --workdir "$BANK/work_${STAMP}_r${RUNG}_${FIRST}_c${CARD}" --out "$ART"
rc=$?
kill $SAMP 2>/dev/null
echo "EXIT rc=$rc $(date -u +%FT%TZ)  banked $ART"
echo "now: $PY perf/land_standing/narrowq_bank.py"
