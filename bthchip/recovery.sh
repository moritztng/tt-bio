#!/usr/bin/env bash
# Board (2,3) under the engine exactly as the booth runs it (stall 120 s, warm 300 s, board reset
# on), so a chip-2 hang is detected, reset and rejoined by the engine itself and timed.
#   recovery.sh RUN_NAME HOURS
# Read: runs/RUN/events.jsonl (chip states for 2 and 3, fold_done), reset.log, chip*.stderr.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
run=$here/runs/$1; mkdir -p "$run"
demo=$HOME/.coworker/wt/bth-chip/demo/sc26
export TT_BIO_LEASE_HOLDER=worker:bth-chip
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
py=$HOME/tt-bio-dev/env/bin/python3
$py "$demo/ops/leases.py" hold 2,3 $$ > "$run/lease.txt"
# qualify_card drives one chip; a later --chips wins, so the engine runs the whole board and
# events.jsonl carries both chips' states.
$py "$demo/ops/qualify_card.py" --chip 2 --hours "$2" --out "$run" --port 8651 --visitor-s 1e9 \
  --server-args "--chips 2,3 --reset-cmd $demo/ops/reset_board.sh --stall-s 120 --warm-s 300 --fold-events $run/fold-events.jsonl" \
  > "$run/qualify.out" 2>&1 &
q=$!
wait $q
echo "qualify rc=$? $(date -u +%FT%TZ)" >> "$run/qualify.out"
$py "$demo/ops/leases.py" release 2,3 >> "$run/lease.txt" 2>&1
