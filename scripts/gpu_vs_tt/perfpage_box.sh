#!/usr/bin/env bash
# One rented box, every perf-page row it owes, unattended. Written for the B200 rental and
# reused verbatim by the H200 and A100 rentals that follow it (TAG picks the column).
#
#   TAG=b200 setsid nohup bash perfpage_box.sh all > /root/results/master.log 2>&1 &
#   TAG=b200 bash perfpage_box.sh setup      # phase 1: install everything, time nothing
#   TAG=b200 bash perfpage_box.sh measure    # phase 2: quiet box, every row
#
# Two rules shape the whole script.
#
# 1. INSTALL EVERYTHING FIRST, MEASURE ON A QUIET BOX. A 25 GB weight pull running next to a
#    timed fold is not the box the published cells were measured on. The host-device-split
#    pass had to throw away boltz-2 and esmfold2 numbers taken at loadavg 25-27; gpu5_session
#    now refuses a row above MAXLOAD, and this script keeps the two phases apart so it never
#    has to.
# 2. EVERY STEP IS IDEMPOTENT. The agent driving this box is a bounded loop that may be
#    relaunched mid-run, so a step whose output already exists is skipped, not repeated. The
#    box is never idle waiting for the agent, and a relaunch never pays twice.
#
# Markers, all under /root/results: SETUP_DONE, MEASURE_DONE, ALL_DONE, plus STEP_<name>.ok
# per step. Poll ALL_DONE from outside; on the last line the log prints the row inventory.
set -uo pipefail

TAG=${TAG:-b200}
R=/root/results
W=/work                      # the design harnesses (bgg, rfd3) are written against /work
REPO=/root/repo              # scripts/gpu_vs_tt + perf/ from the transfer
mkdir -p "$R" "$W"
LOG=$R/master.log

say() { echo "== [$(date -u +%FT%TZ)] $* ==" | tee -a "$LOG"; }

# step <name> <timeout_s> <command...> -- skipped when already ok, marked when it succeeds.
step() {
  local name=$1 tmo=$2; shift 2
  if [ -f "$R/STEP_$name.ok" ]; then say "SKIP $name (already ok)"; return 0; fi
  say "START $name (timeout ${tmo}s)"
  local t0=$SECONDS
  if timeout "$tmo" "$@" >> "$R/step_$name.log" 2>&1; then
    date -u +%FT%TZ > "$R/STEP_$name.ok"
    say "OK $name ($((SECONDS - t0))s)"
  else
    local rc=$?
    echo "$rc" > "$R/STEP_$name.fail"
    say "FAIL $name rc=$rc after $((SECONDS - t0))s -- see step_$name.log, continuing"
  fi
}

quiet_box() {   # wait for the box to go quiet; never measure next to a download
  local i
  for i in $(seq 1 80); do
    awk -v m=4.0 '{exit !($1<m)}' /proc/loadavg && return 0
    say "waiting for a quiet box: loadavg $(cut -d' ' -f1 /proc/loadavg)"
    sleep 30
  done
  say "WARNING box never went quiet; rows will record their own loadavg"
}

# ---------------------------------------------------------------------------- phase 1: setup
do_setup() {
  cd "$REPO/scripts/gpu_vs_tt" || { say "no $REPO -- transfer the harness first"; exit 2; }

  step base 1800 bash gpu5_setup.sh base
  step hostprobe0 600 python3 host_probe.py --out "$R/host_probe_${TAG}_nogpu.json"

  # venv-esm312 is the long pole: 37 GB of weights hang off it (ESMC-6B 25 GB shared by
  # esmfold2/esmfold2-fast, then the six embed repos). Build it first, then pull in the
  # background while every other venv builds in the foreground.
  step esm 2400 bash gpu5_setup.sh esm
  ( step esmweights 5400 bash gpu5_setup.sh esmweights
    step embedweights 5400 bash gpu5_setup.sh embedweights ) &
  BG_ESM=$!
  ( step fetch 3600 bash gpu5_setup.sh fetch ) &
  BG_FETCH=$!

  step boltz 2400 bash gpu5_setup.sh boltz
  step of3 2400 bash gpu5_setup.sh of3

  # The two design harnesses live under /work and bring their own setup scripts.
  mkdir -p "$W"
  for d in scripts perf examples; do
    [ -e "$W/$d" ] || cp -a "$REPO/$d" "$W/$d" 2>/dev/null
  done
  step bgg 3600 bash "$W/scripts/gpu_vs_tt/bgg_setup.sh"
  step rfd3 3600 bash "$W/perf/dsfix/gpu_rfd3_setup.sh"
  # gpu_rfd3_setup.sh installs the code and NOT the 2.51 GB checkpoint; nothing on a detached
  # box can answer the prompt that would normally fetch it, so the A100 pass died at the first
  # design until it pulled it by hand. Do it here, from the URL in foundry's own registry, and
  # refuse a checkpoint whose digest is not the one every other GPU column ran.
  step rfd3ckpt 3600 bash "$REPO/scripts/gpu_vs_tt/rfd3_ckpt.sh"

  say "waiting on the background weight pulls"
  wait $BG_ESM $BG_FETCH
  date -u +%FT%TZ > "$R/SETUP_DONE"
  say "SETUP_DONE"
  df -h / | tee -a "$LOG"
  for v in /root/venv-* /work/venv-* /work/v_head; do
    [ -x "$v/bin/pip" ] && { echo "--- $v"; "$v/bin/pip" freeze 2>/dev/null \
      | grep -Ei 'torch|triton|cuequivariance|boltz|openfold3|transformers|foundry|esm'; }
  done > "$R/pipfreeze_${TAG}.txt" 2>&1
  say "wrote pipfreeze_${TAG}.txt"
}

# ------------------------------------------------------------------- phase 2: the rows
# The audit arm runs FIRST. It is the one thing Moritz asked for by name, and it needs only
# the two cheap venvs, so it lands even if a later stage eats the rental.
do_measure() {
  cd "$REPO/scripts/gpu_vs_tt" || exit 2
  quiet_box
  step hostprobe 900 /root/venv-boltz/bin/python host_probe.py --out "$R/host_probe_${TAG}.json"

  # --- audit rows, published protocol, published batch ------------------------------------
  step audit_boltz 2400 env TAG="$TAG" MODELS="boltz-2" bash gpu5_session.sh
  step audit_of3   2400 env TAG="$TAG" MODELS="openfold3" bash gpu5_session.sh
  step audit_bgg   3600 "$W/venv-bgg/bin/python" "$W/scripts/gpu_vs_tt/bgg_bench.py" \
       "${TAG^^}" headline
  # batch 1 is the published cell (matched to the p150a arm); --n-batches defaults to 4, so
  # the cold batch is discarded and n=3 warm, exactly as published.
  step audit_rfd3  3600 env TAG="$TAG" bash rfd3_prod_run.sh 1

  # The counter says cuEquivariance ran; only a per-call timing at the model's own shape says
  # WHICH kernel ran, which is the entire cu12/cu13 question on Blackwell.
  step probe_cu12 900 /root/venv-boltz/bin/python cueq_tri_probe.py \
    --from-json "$R/gpu_boltz-2_prot512_${TAG}.json" --out "$R/cueq_probe_cu12_${TAG}.json"

  # --- the phase-decomposition arms: is the loss in the trunk or in the sampler? -----------
  # T(recycles, steps) at four points solves for per-recycle and per-step cost without a
  # profiler. Every published row the B200 WINS runs 10 recycles; every row it loses runs 3 or
  # is a design rollout. That is either a real work-per-kernel effect or a coincidence across
  # 8 rows, and these four points plus the same four on the H200 tell them apart.
  for rs in "10 200" "3 50" "10 50"; do
    set -- $rs
    step "phase_boltz_r$1_s$2" 2400 /root/venv-boltz/bin/python gpu5_bench.py \
      --model boltz-2 --repeat 3 --power --recycles "$1" --steps "$2" \
      --yaml "$REPO/perf/size512/fixtures/cdk2x2_512.yaml" \
      --a3m "$REPO/perf/size512/fixtures/cdk2x2_512.a3m" --work /root/work \
      --out "$R/phase_boltz-2_r$1_s$2_${TAG}.json"
  done

  # --- the cu13 question, boltz-2 and boltzgen only ---------------------------------------
  # cuEquivariance 0.8.0+ ships Blackwell (CC 10.0/10.3) BF16/FP16 triangle-attention kernels
  # and 0.10.0 an sm100f forward kernel, hidden_dim <= 256 -- "only available on cu13 builds"
  # (NVIDIA changelog). boltz-2 and boltzgen both run bf16, so both are eligible and both were
  # published on the cu12 ops wheel. OpenFold3 (fp32) and RFdiffusion3 (no cuEquivariance at
  # all) are not eligible and are the controls. This is an ALTERNATE arm, never the cell.
  step cu13_wheels 1800 bash cu13_flip.sh
  step probe_cu13 900 /root/venv-boltz/bin/python cueq_tri_probe.py \
    --from-json "$R/gpu_boltz-2_prot512_${TAG}.json" --out "$R/cueq_probe_cu13_${TAG}.json"
  step audit_boltz_cu13 2400 env TAG="${TAG}cu13" MODELS="boltz-2" bash gpu5_session.sh
  step audit_bgg_cu13 3600 "$W/venv-bgg/bin/python" "$W/scripts/gpu_vs_tt/bgg_bench.py" \
       "${TAG^^}CU13" headline

  # --- host sensitivity: does core count move a dispatch-bound row on THIS box? ------------
  for n in 12 6; do
    step "cores$n" 2400 taskset -c "0-$((n - 1))" /root/venv-boltz/bin/python gpu5_bench.py \
      --model boltz-2 --repeat 3 --power \
      --yaml "$REPO/perf/size512/fixtures/cdk2x2_512.yaml" \
      --a3m "$REPO/perf/size512/fixtures/cdk2x2_512.a3m" --work /root/work \
      --out "$R/cores${n}_boltz-2_${TAG}.json"
  done

  # --- the seven new rows -----------------------------------------------------------------
  step new_esmfold2fast 3600 env TAG="$TAG" MODELS="esmfold2-fast" bash gpu5_session.sh
  for m in esmc-300m esmc-600m esmc-6b saprot-35m saprot-650m saprot-1.3b; do
    step "new_$m" 2400 /root/venv-esm312/bin/python gpu_embed_bench.py --model "$m" \
      --repeat 3 --out "$R/gpu_embed_${m}_prot512_${TAG}.json"
  done

  date -u +%FT%TZ > "$R/MEASURE_DONE"
  say "MEASURE_DONE"
}

case "${1:-all}" in
  setup)   do_setup ;;
  measure) do_measure ;;
  all)     do_setup; do_measure ;;
  *) echo "usage: $0 {setup|measure|all}" >&2; exit 2 ;;
esac

say "INVENTORY"
ls -l "$R"/*.json "$R"/*.jsonl 2>/dev/null | tee -a "$LOG"
ls "$R"/STEP_*.fail 2>/dev/null | tee -a "$LOG"
date -u +%FT%TZ > "$R/ALL_DONE"
say "ALL_DONE"
