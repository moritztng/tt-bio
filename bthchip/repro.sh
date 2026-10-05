#!/usr/bin/env bash
# Chip 2 alone on the booth workload, engine reset disabled, hangwatch beside it.
#   repro.sh RUN_NAME HOURS [ENGINE_DIR] [AICLK_MHZ]
# ENGINE_DIR defaults to the deployed ~/sc26/demo/sc26. AICLK_MHZ pins chip 2's clock for the run
# (perf/pvxcust/pin_aiclk.py, released when the run ends). On engines with --hang-s, tt-metal's own
# dispatch timeout runs tt-triage on the hung process; the run then stops at that first hang
# (SIGINT to qualify_card), so no second worker is started on a hung chip.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
run=$here/runs/$1; mkdir -p "$run/metal"
demo=${3:-$HOME/sc26/demo/sc26}
clk=${4:-}
site=$(ls -d "$HOME"/tt-bio-dev/env/lib/python3*/site-packages)
export TT_BIO_LEASE_HOLDER=worker:bth-chip TT_METAL_LOGS_PATH=$run/metal TT_METAL_INSPECTOR=1
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="cd $run && TT_VISIBLE_DEVICES=2 timeout 600 $here/triage-venv/bin/python $site/triage/triage.py --skip-version-check --disable-progress --disable-colors -v > $run/triage-\$(date -u +%H%M%S).txt 2>&1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
py=$HOME/tt-bio-dev/env/bin/python3
$py "$demo/ops/leases.py" hold 2 $$ > "$run/lease.txt"
pin=
if [ -n "$clk" ]; then
  $py "$here/../perf/pvxcust/pin_aiclk.py" 2 "$clk" > "$run/pin.log" 2>&1 &
  pin=$!
  sleep 3
fi
python3 "$here/hangwatch.py" --port 8641 --out "$run" --logs "$run/metal" --quiet-s 60 > "$run/hangwatch.log" 2>&1 &
$py "$demo/ops/qualify_card.py" --chip 2 --hours "$2" --out "$run" --port 8641 --visitor-s 1e9 \
  --server-args "--long-res 100000 --stall-s 100000 --warm-s 100000 --fold-events $run/fold-events.jsonl" > "$run/qualify.out" 2>&1 &
q=$!
while kill -0 $q 2>/dev/null; do
  if grep -qs "TIMEOUT: device timeout\\|chip_hung" "$run/chip2.stderr" "$run/server.log"; then
    echo "$(date -u +%FT%TZ) first hang seen, stopping the run" >> "$run/qualify.out"
    # qualify_card ignores SIGINT (a background job of this script); the engine does not, and
    # qualify_card writes its summary once the engine exits.
    pkill -INT -f "engine/server.py --chips 2 --port 8641"; wait $q; break
  fi
  sleep 5
done
wait $q
echo "qualify rc=$? $(date -u +%FT%TZ)" >> "$run/qualify.out"
[ -n "$pin" ] && kill -INT $pin && wait $pin
$py "$demo/ops/leases.py" release 2 >> "$run/lease.txt" 2>&1
