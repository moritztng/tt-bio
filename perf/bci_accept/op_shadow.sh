#!/bin/bash
# The per-op taped/shipped shadow diff on a card, under state/bci/CHIPS.md's RULES.
#
#   bash perf/bci_accept/op_shadow.sh <chip> [tokens]
#
# Opens a device. Run it only when the NOW block of state/bci/CHIPS.md names bci-accept as the
# holder of <chip>. The flock is what keeps one BCI device job on a card at a time; do not
# bypass it. Never `tt-smi -r` from here.
set -u
CARD=${1:?chip number from CHIPS.md NOW}
TOKENS=${2:-64}
# shellcheck source=perf/bci_accept/host_profile.sh
. "$(dirname "${BASH_SOURCE[0]}")/host_profile.sh"
OUT=$WT/.bci/op_shadow_chip${CARD}_t${TOKENS}${TAG}.log
CLK=$WT/.bci/op_shadow_chip${CARD}_t${TOKENS}${TAG}.aiclk

cd "$WT" || exit 1
[ -e /home/ttuser/fdv_perf_quiet ] && { echo "fdv_perf_quiet exists: FDV is measuring, not starting" >&2; exit 1; }

export PYTHONPATH=/home/ttuser/bcx_e2e/bc2:$WT
export TT_VISIBLE_DEVICES=$CARD

# shellcheck disable=SC2206
EXTRA=(${BCI_EXTRA:-})

# Sampled DURING the run: a clock read while the chip is idle says nothing about the clock the
# work ran at, and 4294967295 is a dead ARC's sentinel rather than a reading.
sample_aiclk() {
  while :; do
    printf '%s %s\n' "$(date -u +%H:%M:%SZ)" \
      "$(cat "/sys/class/tenstorrent/tenstorrent!$CLK_NODE/tt_aiclk" 2>/dev/null || echo unread)"
    sleep 10
  done > "$CLK"
}
sample_aiclk &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

{
  echo "=== per-op shadow diff, chip $CARD, $TOKENS tokens ${BCI_EXTRA:-} ==="
  echo "SRC_REV $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  date -u +'start %Y-%m-%dT%H:%M:%SZ'
  flock -w "$LOCK_WAIT" "$LOCK" \
    nice -n 10 timeout 2400 "$PY" perf/bci_accept/op_shadow_diff.py \
      --card "$CARD" --af2-weights "$AF2" --tokens "$TOKENS" ${EXTRA[@]+"${EXTRA[@]}"}
  echo "=== rc=$? ==="
  date -u +'end %Y-%m-%dT%H:%M:%SZ'
} > "$OUT" 2>&1

kill $SAMPLER 2>/dev/null
echo "wrote $OUT, AICLK samples in $CLK"
