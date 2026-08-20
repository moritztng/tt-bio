#!/usr/bin/env bash
# Drive the AF2-IG host-CPU ladder, one benchlock acquisition per rung.
#
# One rung per lock, not one lock for the campaign: qb2 runs several timed device A/Bs in
# parallel by design and this ladder takes hours, so holding the box for the whole thing
# would stall the fleet. Each rung is its own invocation anyway, which is also what
# upstream does (fresh subprocess per AF2 call), so the fixed term is measured per rung
# exactly as production pays it.
set -u
WT=/home/ttuser/.coworker/wt/pxdesign-af2ig-decision
PY=/home/ttuser/pxd_af2_cpu/bin/python
LOGS=/home/ttuser/pxd_bench_logs
BL=/home/ttuser/.coworker/scripts/benchlock.sh
cd "$WT" || exit 1
mkdir -p "$LOGS"

run_rung() {
  local tgt="$1" nd="$2"
  local out="perf/pxdesign/af2ig_cpu_t${tgt}.json"
  local log="$LOGS/af2ig_rung_${tgt}.log"
  if [ -f "$out" ] && grep -q '"marginal_s_per_design": [0-9]' "$out"; then
    echo "[ladder] rung $tgt already has a marginal, skipping" >> "$LOGS/ladder.log"
    return 0
  fi
  echo "[ladder] $(date -Is) starting rung $tgt ndesign=$nd" >> "$LOGS/ladder.log"
  BENCHLOCK_WAIT_S=7200 "$BL" worker:pxdesign-af2ig-decision -- \
      "$PY" -u perf/pxdesign/af2ig_cpu_bench.py \
      --target "$tgt" --ndesign "$nd" --out "$out" > "$log" 2>&1
  echo "[ladder] $(date -Is) rung $tgt exit=$?" >> "$LOGS/ladder.log"
}

# Smallest first: the cheap rungs validate the instrument and give the exponent before the
# expensive ones are spent on.
#
# One design per rung would already give the marginal to within 1.3 %: rung 128 measured the
# fixed term (params load + XLA compile) at 8.6 s against a 657.0 s marginal, so cold ~= marginal.
# Two designs on the upper rungs buys a same-shape repeat rather than a tighter marginal.
#
# The 768 rung (848 tokens) is deliberately NOT run. Rung 128 already puts AF2-IG-on-host at 244x
# over its own 4x bar, and the H200 is only 5.6 % utilised at that rung, so the small rung is the
# one that FLATTERS the CPU -- the gap can only widen with size. Ten hours of a shared box to
# refine a number that cannot change the verdict is the same call the brief already makes for the
# Protenix term in Phase 0 step 4. 848 is reported as extrapolated, labelled as such.
run_rung 128 3
run_rung 256 2
run_rung 512 2
echo "[ladder] $(date -Is) ALL RUNGS DONE" >> "$LOGS/ladder.log"
