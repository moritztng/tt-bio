#!/bin/bash
# The whole remaining RFD3_TUNE_MATMUL verification, as ONE command.
#
# p16 lost two turns to the same shape of failure: the measurement sequence is four separate
# device runs, each one long enough that a turn boundary or a contended card leaves the pass
# half-done and the next turn re-derives where it got to. This runs all four to completion and
# writes one report, so the default decision either has its full evidence or visibly does not.
#
#   scripts/rfd3_port/run_p16_verification.sh [--rounds N]
#
# Waits for the card first. Two different waits, for two different reasons:
#
#  * A release gate (scripts/full_parity_gate.py, scripts/release_gate.py) is a MANY-LEG run that
#    opens and closes the card between legs. Merely waiting on the lease would let this script
#    win the gap between two of its legs and fail the gate's next leg with DeviceInUseError --
#    a release outranks a perf pass, so wait for the gate PROCESS to be gone, not for the lease.
#  * Against everything else, TT_BIO_LEASE_TIMEOUT (device_lease.py reads it; default 120s) does
#    the right thing: block instead of dying. 120s is far shorter than one bench run, so a
#    transient neighbour used to cost the whole sweep.
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
# Defaults are pc/card-0; PY, TT_VISIBLE_DEVICES and TT_BIO_LEASE_HOLDER are all overridable
# because the measurement needs one free card and does not care which host it is on. p16 lost a
# whole pass waiting on pc while three other cards sat idle.
PY=${PY:-/home/moritz/tt-bio/env/bin/python3}
REPORT=$WT/scripts/rfd3_port/p16_verification_report.txt
ROUNDS=2
[ "${1:-}" = --rounds ] && { ROUNDS=$2; shift 2; }

cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0}
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:rfd3-tune-matmul-sweep}
export TT_BIO_LEASE_TIMEOUT=${TT_BIO_LEASE_TIMEOUT:-3600}
export TT_METAL_LOGGER_LEVEL=FATAL
export LOGURU_LEVEL=WARNING

export PY

say() { echo "$*" | tee -a "$REPORT"; }

: > "$REPORT"
say "=== p16 RFD3_TUNE_MATMUL verification  start $(date -Is) host=$(hostname) ==="

# --- 0. yield to any release gate ------------------------------------------------------------
# Match on argv STRUCTURE (argv[0] is a python, argv[1] is the gate script), not on `pgrep -f`
# over the whole command line: the release-gate worker's own monitoring shell embeds the gate's
# path in a pgrep of its own, so a command-line substring match reports a gate that is really a
# shell watching one, and this loop would wait on a process that never opens the card.
gate_pids() {
  ps -eo pid,args --no-headers 2>/dev/null | awk \
    '$2 ~ /python3?$/ && $3 ~ /(full_parity_gate|release_gate|protenix_ife_parity_gate)\.py$/ {print $1}'
}
if [ -n "$(gate_pids)" ]; then
  say "waiting for release gate to finish: $(gate_pids | tr '\n' ' ')"
  while [ -n "$(gate_pids)" ]; do sleep 30; done
  say "release gate cleared at $(date -Is); settling 20s before opening the card"
  sleep 20
fi

# --- 1. trace-OFF (shipped) A/B, the gate on flipping the default ----------------------------
say ""
say "--- [1/4] shipped-config A/B (TRACE=0), $ROUNDS rounds x 5 fixtures ---"
TRACE=0 "$WT/scripts/rfd3_port/run_tune_matmul_sweep.sh" --rounds "$ROUNDS" \
    iai40 iai80 iai150 mpro iai250 >>"$REPORT" 2>&1
say "sweep rc=$? (rows also appended to scripts/rfd3_port/tune_matmul_sweep.log)"

# --- 2. trace-ON A/B, cross-check that the win is not one dispatch regime's artifact ---------
# Not part of the decision rule -- [1] alone decides the default -- and it costs ~20 min of card,
# so CROSSCHECK=0 skips it when the card is the scarce resource.
if [ "${CROSSCHECK:-1}" = 1 ]; then
say ""
say "--- [2/4] trace-ON A/B cross-check (TRACE=1), $ROUNDS rounds x iai40 iai150 ---"
TRACE=1 "$WT/scripts/rfd3_port/run_tune_matmul_sweep.sh" --rounds "$ROUNDS" \
    iai40 iai150 >>"$REPORT" 2>&1
say "sweep rc=$?"
else
say ""
say "--- [2/4] trace-ON A/B cross-check SKIPPED (CROSSCHECK=0) ---"
fi

# --- 3. first-design calibration overhead ----------------------------------------------------
# bench_batch_designs_per_sec.py CANNOT see this: it warms up with a full 4-step sample, so the
# first-call calibration compiles land in the warmup, outside its timed region. Measure it the
# way a user pays it -- whole `tt-bio design` wall clock, cold process, D=8 (num_designs 8 at the
# default batch_size 8 is one batch). At 5 timesteps the fixed calibration cost is spread over
# few steps and at 20 it is spread over many, so the pair says whether default-on needs an
# n_timesteps floor.
say ""
say '--- [3/4] calibration overhead: tt-bio design wall clock, D=8, cold process ---'
SPEC=$WT/scripts/rfd3_port/parity_artifacts/iai_protein/iai_inputs.yaml
for steps in 5 20; do
  for round in $(seq 1 "$ROUNDS"); do
    # alternate order per round: whichever runs second inherits a warmer card (~5%, p16 defect 3)
    order="0 1"; [ $((round % 2)) -eq 0 ] && order="1 0"
    for tune in $order; do
      out=$(mktemp -d); t0=$(date +%s.%N)
      RFD3_TUNE_MATMUL=$tune PYTHONPATH="$WT" "$PY" -m tt_bio.main design "$SPEC" \
          --from_pdb --out_dir "$out" --num_designs 8 --num_timesteps "$steps" --devices "$TT_VISIBLE_DEVICES" \
          >"$out/log" 2>&1
      rc=$?; t1=$(date +%s.%N)
      say "[calib steps=$steps r$round tune=$tune] wall=$(echo "$t1 - $t0" | bc)s rc=$rc \
cifs=$(find "$out" -name '*.cif' 2>/dev/null | wc -l)"
      [ $rc -ne 0 ] && { say "  FAILED, last 12 lines:"; tail -12 "$out/log" | tee -a "$REPORT"; }
      rm -rf "$out"
    done
  done
done

# --- 4. release perf leg ---------------------------------------------------------------------
# num_designs=1/num_timesteps=4 in perf_regression.py's rfd3 SPEC means D=1, and _tunable
# requires xs[0] > 1, so the tuned path is inert here BY CONSTRUCTION. This leg therefore cannot
# regress from the flag and cannot demonstrate its win either; it runs to prove the first half.
say ""
say "--- [4/4] release perf leg (D=1, tuned path inert by construction) ---"
for tune in 0 1; do
  RFD3_TUNE_MATMUL=$tune PYTHONPATH="$WT" "$PY" "$WT/scripts/perf_regression.py" --model rfd3 \
      >>"$REPORT" 2>&1
  say "perf_regression --model rfd3 (RFD3_TUNE_MATMUL=$tune) rc=$?"
done

say ""
say "=== done $(date -Is) ==="
say "Decision rule: flip _TUNE_MATMUL default-on ONLY if [1] is a win at every fixture at D=8."
say "If [3] shows steps=5 net-negative but steps=20 positive, gate default-on on n_timesteps."
