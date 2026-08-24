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

# Wait until THIS CONTAINER is quiet, and until no stranger is on the GPU.
#
# /proc/loadavg is NOT namespaced: on a 192-core shared host it reports the whole machine's
# runnable count, which on the B200 box sat at 15-31 while our own processes used 0.0% CPU.
# A "loadavg < 4" gate can therefore never pass there and just burns its own timeout. What the
# rule actually means is "do not measure next to your own 25 GB download", so measure our own
# cgroup CPU draw instead, which is the thing we control and the thing that perturbs a row.
# The GPU check is the other half: a co-tenant's process on the card invalidates a row outright
# (a stranger's 12486 MiB once turned a <4% spread into 9.7-15.2 s).
own_cores() {
  local a b
  a=$(awk '/usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)
  sleep 5
  b=$(awk '/usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)
  awk -v a="$a" -v b="$b" 'BEGIN{printf "%.2f", (b-a)/5000000.0}'
}
foreign_gpu() {   # count compute processes on the card that are not ours
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true
}
quiet_box() {
  local i c g
  for i in $(seq 1 40); do
    c=$(own_cores); g=$(foreign_gpu)
    if awk -v c="$c" 'BEGIN{exit !(c<2.0)}' && [ "${g:-0}" -eq 0 ]; then
      say "box quiet: own cgroup draw ${c} cores, no GPU compute apps (host loadavg $(cut -d' ' -f1 /proc/loadavg), not namespaced)"
      return 0
    fi
    say "waiting: own draw ${c} cores, foreign GPU procs ${g} (host loadavg $(cut -d' ' -f1 /proc/loadavg), not namespaced)"
    sleep 25
  done
  say "WARNING never went quiet; rows record their own loadavg and their own cgroup draw"
}

# ---------------------------------------------------------------------------- phase 1: setup
do_setup() {
  cd "$REPO/scripts/gpu_vs_tt" || { say "no $REPO -- transfer the harness first"; exit 2; }

  step base 1800 bash gpu5_setup.sh base
  step hostprobe0 600 python3 host_probe.py --out "$R/host_probe_${TAG}_image.json"

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

  # Nesso-1 landed on main after the manifest froze (merge c902ecdf), and the A100 rental is the
  # only box in the campaign that can still give it a GPU column. SETUP_NESSO=1 builds /work/v_nesso
  # and prefetches its three assets (ccd.pkl, model.safetensors, ESM-2 650M). FOREGROUND and after
  # of3 on purpose: gpu_nesso1_setup.sh runs apt-get, so does stage_of3, and two concurrent apts
  # deadlock on the dpkg lock. It also writes /work/{setup.log,SETUP_OK,SETUP_FAIL}, the same three
  # paths gpu_rfd3_setup.sh uses -- safe only because SETUP_DESIGN=0 on this box.
  if [ "${SETUP_NESSO:-0}" = 1 ]; then
    step nesso 3600 bash "$W/perf/nesso1/gpu_nesso1_setup.sh"
  fi

  # The two design harnesses live under /work and bring their own setup scripts. SETUP_DESIGN=0
  # skips both: they cost a venv, 3.9 GB of BoltzGen weights and a 2.51 GB RFdiffusion3
  # checkpoint, and a box that owes no design row should not pay for them. The H200 rental owes
  # none -- both design cells are published there and are do-not-remeasure.
  mkdir -p "$W"
  for d in scripts perf examples; do
    [ -e "$W/$d" ] || cp -a "$REPO/$d" "$W/$d" 2>/dev/null
  done
  if [ "${SETUP_DESIGN:-1}" != 1 ]; then
    say "SETUP_DESIGN=0: skipping bgg / rfd3 / rfd3ckpt, this box owes no design row"
  else
    step bgg 3600 bash "$W/scripts/gpu_vs_tt/bgg_setup.sh"
    # gpu_rfd3_setup.sh execs its own stdout to /work/setup.log, so step_rfd3.log stays empty and
    # /work/{setup.log,SETUP_OK,SETUP_FAIL} is where its story is.
    step rfd3 3600 bash "$W/perf/dsfix/gpu_rfd3_setup.sh"
    # gpu_rfd3_setup.sh installs the code and NOT the 2.51 GB checkpoint; nothing on a detached
    # box can answer the prompt that would normally fetch it, so the A100 pass died at the first
    # design until it pulled it by hand. Do it here, from the URL in foundry's own registry, and
    # refuse a checkpoint whose digest is not the one every other GPU column ran.
    step rfd3ckpt 3600 bash "$REPO/scripts/gpu_vs_tt/rfd3_ckpt.sh"
  fi

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

  # --- the seven new rows -----------------------------------------------------------------
  step new_esmfold2fast 3600 env TAG="$TAG" MODELS="esmfold2-fast" bash gpu5_session.sh
  # EMBED_REPEAT=15, not 3. Every one of these six rows is 7-41 ms of device work, so with only
  # three samples the max-min spread is host scheduling noise rather than a property of the GPU:
  # on the B200, saprot-650m read 58.1 % at n=3 and 3.4 % at n=15, while its median moved 9 %.
  # The medians agreed to 5-9 % between the two, so n=3 was not wrong, it was just imprecise on
  # rows this short. Still inside the protocol's "n>=3 where affordable", and these rows cost
  # seconds each.
  for m in esmc-300m esmc-600m esmc-6b saprot-35m saprot-650m saprot-1.3b; do
    step "new_$m" 2400 /root/venv-esm312/bin/python gpu_embed_bench.py --model "$m" \
      --repeat "${EMBED_REPEAT:-15}" --out "$R/gpu_embed_${m}_prot512_${TAG}.json"
  done

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

  # --- host sensitivity: does core count move a dispatch-bound row on THIS box?
  # Before the cu13 swap on purpose: the 24-core point of this ladder is the published-arm run
  # above, and a ladder measured across two different kernel stacks measures the stack. ------------
  for n in 12 6; do
    step "cores$n" 2400 taskset -c "0-$((n - 1))" /root/venv-boltz/bin/python gpu5_bench.py \
      --model boltz-2 --repeat 3 --power \
      --yaml "$REPO/perf/size512/fixtures/cdk2x2_512.yaml" \
      --a3m "$REPO/perf/size512/fixtures/cdk2x2_512.a3m" --work /root/work \
      --out "$R/cores${n}_boltz-2_${TAG}.json"
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

  date -u +%FT%TZ > "$R/MEASURE_DONE"
  say "MEASURE_DONE"
}

# ------------------------------------------------------- phase 2, H200: control first, then rows
# The H200 rental owes a different set than the B200 one, and the difference is not cosmetic:
#
#   * H200 is the perf page's INDEX platform. DGX H200 = 1.00x in the cost model and the
#     "beat the Nvidia server" bar is derived from an H200 number, so a cell that is not
#     comparable to the published column moves the bar every other row is judged against.
#     Hence a mandatory harness control reproducing an already-published cell (ESMFold2,
#     7.256 s, the cheapest of the eight) BEFORE any new row is believed.
#   * No audit arm. It is answered: the B200 pass reproduced all four contested cells within
#     4.1 % on a config-identical machine, so the H200's lead is real.
#   * No design rows and no cu13 arm. Both design cells are published here and the Blackwell
#     triangle-attention wheel question does not exist on sm_90.
#   * Three arms the B200 pass could only half-measure, each cheap here and each needed on both
#     sides before its conclusion is symmetric: host_probe, the four boltz-2 (recycles, steps)
#     points, and power on a trunk-heavy and a sampler-heavy row.
do_measure_h200() {
  cd "$REPO/scripts/gpu_vs_tt" || exit 2
  quiet_box
  step hostprobe 900 /root/venv-boltz/bin/python host_probe.py --out "$R/host_probe_${TAG}.json"

  # --- the control. TAG is ${TAG}ctl so its JSON can never be mistaken for, or overwrite, the
  # published cell's file name. It is a check, not a replacement. -----------------------------
  step control_esmfold2 3600 env TAG="${TAG}ctl" MODELS="esmfold2" bash gpu5_session.sh
  step control_verdict 300 python3 control_verdict.py \
    --result "$R/gpu_esmfold2_prot512_${TAG}ctl.json" --published 7.256 \
    --label "floating venv-esm312, esm@main as stage_esm resolves it" \
    --out "$R/CONTROL_VERDICT_${TAG}.json"

  # Anything but COMPARABLE and the pinned arm runs, unattended, right here. stage_esm installs
  # esm from git @main and lets torch resolve; the published cell ran esm 3.3.0 @26b0bc2b and
  # torch 2.13.0+cu130, and the B200 box resolved esm 3.4.0 and torch 2.11.0+cu130 four days
  # later. So package drift is a named, live alternative to "the box or the harness is wrong",
  # and this is what tells them apart without the agent diagnosing it while the meter runs.
  if [ -f "$R/CONTROL_RERUN_WANTED" ]; then
    say "control missed its band -- running the pinned-package control arm"
    step esmctl 2400 bash gpu5_setup.sh esmctl
    step control_esmfold2_pinned 3600 env TAG="${TAG}ctlpin" ESM_VENV=/root/venv-esm312ctl \
      MODELS="esmfold2" bash gpu5_session.sh
    step control_verdict_pinned 300 python3 control_verdict.py \
      --result "$R/gpu_esmfold2_prot512_${TAG}ctlpin.json" --published 7.256 \
      --label "pinned venv-esm312ctl, esm@26b0bc2b + transformers 4.57.6" \
      --out "$R/CONTROL_VERDICT_PINNED_${TAG}.json"
  fi

  # --- the seven new rows. The control does NOT gate these: they are ~15 min of a rental whose
  # cost is the box, and stopping the box to think would mean re-renting to get them. What the
  # control governs is whether the doc may call them comparable to the published column, which
  # is a publication decision, not a measurement one. -----------------------------------------
  step new_esmfold2fast 3600 env TAG="$TAG" MODELS="esmfold2-fast" bash gpu5_session.sh
  for m in esmc-300m esmc-600m esmc-6b saprot-35m saprot-650m saprot-1.3b; do
    step "new_$m" 2400 /root/venv-esm312/bin/python gpu_embed_bench.py --model "$m" \
      --repeat "${EMBED_REPEAT:-15}" --out "$R/gpu_embed_${m}_prot512_${TAG}.json"
  done

  # --- the three handed-off arms, deliverables already in hand -------------------------------
  # T(recycles, steps) = base + a*recycles + b*steps at four points, solved without a profiler.
  # The B200 side reads 0.541 s per recycle and 0.0288 s per step, i.e. 70 % of its published
  # cell is the 200-step sampler, which is the mechanism behind the perfect split of all eight
  # rows by recycle count. The same four points here turn that into a ratio. (3,200) is the
  # published boltz-2 configuration, so it doubles as a second free control on this box.
  for rs in "3 200" "10 200" "3 50" "10 50"; do
    set -- $rs
    step "phase_boltz_r$1_s$2" 2400 /root/venv-boltz/bin/python gpu5_bench.py \
      --model boltz-2 --repeat 3 --power --recycles "$1" --steps "$2" \
      --yaml "$REPO/perf/size512/fixtures/cdk2x2_512.yaml" \
      --a3m "$REPO/perf/size512/fixtures/cdk2x2_512.a3m" --work /root/work \
      --out "$R/phase_boltz-2_r$1_s$2_${TAG}.json"
  done
  # A counter proves cuEquivariance ran; only a per-call timing at the model's own recorded
  # shape says what it cost. Same probe the B200 ran, so the two are comparable per call.
  step probe_cueq 900 /root/venv-boltz/bin/python cueq_tri_probe.py \
    --from-json "$R/phase_boltz-2_r3_s200_${TAG}.json" --out "$R/cueq_probe_${TAG}.json"
  # OpenFold3 is the second-largest of the four contested rows and the B200 read it at 33.8 % of
  # a 1000 W limit and 54 % utilisation. The saturation claim needs both sides. TAG ...sat so the
  # file cannot be confused with the published openfold3 cell.
  step sat_of3 2400 env TAG="${TAG}sat" MODELS="openfold3" bash gpu5_session.sh
  # Last, and droppable: the host-sensitivity derivative. The B200 side is flat (8.285 s full,
  # 8.163 s on 12 cores, 8.192 s on 6), which already refutes the host confound; this makes the
  # statement symmetric on the box that has the wider host.
  for n in 12 6; do
    step "cores$n" 2400 taskset -c "0-$((n - 1))" /root/venv-boltz/bin/python gpu5_bench.py \
      --model boltz-2 --repeat 3 --power \
      --yaml "$REPO/perf/size512/fixtures/cdk2x2_512.yaml" \
      --a3m "$REPO/perf/size512/fixtures/cdk2x2_512.a3m" --work /root/work \
      --out "$R/cores${n}_boltz-2_${TAG}.json"
  done

  date -u +%FT%TZ > "$R/MEASURE_DONE"
  say "MEASURE_DONE"
}

# ------------------------------------------------- phase 2, A100: control first, then rows, then Nesso
# The A100 rental is the last of the three and owes the same seven rows, but three things differ:
#
#   * The published A100 column was measured on machine 38441, and an offer on that exact machine
#     is available again, so the control is a test of the harness and the package stack rather than
#     of the hardware -- the same position the H200 pass was in. Published cell: ESMFold2 14.741 s.
#   * sm_80 is the least-proven generation for cuEquivariance in this campaign, so every counter is
#     reported either way. The prior A100 pass found the ops-cu12 wheels ship native sm_80 cubins and
#     every counter matched H200/B200 exactly, and the NVIDIA changelog gates only CC 10.0/10.3
#     kernels behind cu13 -- so there is nothing to flip here and a fallback would be a harness bug.
#   * Nesso-1 merged to main after the freeze (c902ecdf) and has no A100 number anywhere. It runs
#     LAST and is droppable: an extra row must never delay a frozen one.
#
# No design rows (both A100 design cells are published), no cu13 arm (nothing on sm_80 is cu13-gated),
# no audit arm (the B200 pass answered it).
do_measure_a100() {
  cd "$REPO/scripts/gpu_vs_tt" || exit 2
  # The SKU is part of the cell, so make it an artifact rather than terminal scrollback. A100 SXM4
  # 80 GB reads a 400 W limit and 81920 MiB; the PCIe parts read 250/300 W and the 40 GB SXM4 reads
  # 40960 MiB. Anything else and this box is the wrong part.
  step sku 120 bash -c "nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,power.max_limit,clocks.max.sm,compute_cap --format=csv > $R/sku_${TAG}.txt; cat /sys/fs/cgroup/cpu.max >> $R/sku_${TAG}.txt; grep -m1 'model name' /proc/cpuinfo >> $R/sku_${TAG}.txt; nproc >> $R/sku_${TAG}.txt; cat $R/sku_${TAG}.txt"
  quiet_box
  step hostprobe 900 /root/venv-boltz/bin/python host_probe.py --out "$R/host_probe_${TAG}.json"

  # --- the control, against the published A100 ESMFold2 cell. TAG=${TAG}ctl so its JSON can never
  # be mistaken for, or overwrite, a published cell's file name. A check, not a replacement. ------
  step control_esmfold2 3600 env TAG="${TAG}ctl" MODELS="esmfold2" bash gpu5_session.sh
  step control_verdict 300 python3 control_verdict.py \
    --result "$R/gpu_esmfold2_prot512_${TAG}ctl.json" --published 14.741 \
    --label "floating venv-esm312, esm@main as stage_esm resolves it" \
    --out "$R/CONTROL_VERDICT_${TAG}.json"
  if [ -f "$R/CONTROL_RERUN_WANTED" ]; then
    say "control missed its band -- running the pinned-package control arm"
    step esmctl 2400 bash gpu5_setup.sh esmctl
    step control_esmfold2_pinned 3600 env TAG="${TAG}ctlpin" ESM_VENV=/root/venv-esm312ctl \
      MODELS="esmfold2" bash gpu5_session.sh
    step control_verdict_pinned 300 python3 control_verdict.py \
      --result "$R/gpu_esmfold2_prot512_${TAG}ctlpin.json" --published 14.741 \
      --label "pinned venv-esm312ctl, esm@26b0bc2b + transformers 4.57.6" \
      --out "$R/CONTROL_VERDICT_PINNED_${TAG}.json"
  fi

  # --- the seven frozen rows. The control does NOT gate them: they are minutes of a rental whose
  # cost is the box. What the control governs is whether the doc may call them comparable to the
  # published column, which is a publication decision, not a measurement one. -------------------
  step new_esmfold2fast 3600 env TAG="$TAG" MODELS="esmfold2-fast" bash gpu5_session.sh
  for m in esmc-300m esmc-600m esmc-6b saprot-35m saprot-650m saprot-1.3b; do
    step "new_$m" 2400 /root/venv-esm312/bin/python gpu_embed_bench.py --model "$m" \
      --repeat "${EMBED_REPEAT:-15}" --out "$R/gpu_embed_${m}_prot512_${TAG}.json"
  done

  # --- the third side of the campaign's two-sided arms. Both are already measured on B200 and H200;
  # the A100 turns the trunk/sampler split and the host probe into a three-card statement. (3,200) is
  # the published A100 boltz-2 configuration (14.395 s), so it is a second free control. -----------
  for rs in "3 200" "10 200" "3 50" "10 50"; do
    set -- $rs
    step "phase_boltz_r$1_s$2" 2400 /root/venv-boltz/bin/python gpu5_bench.py \
      --model boltz-2 --repeat 3 --power --recycles "$1" --steps "$2" \
      --yaml "$REPO/perf/size512/fixtures/cdk2x2_512.yaml" \
      --a3m "$REPO/perf/size512/fixtures/cdk2x2_512.a3m" --work /root/work \
      --out "$R/phase_boltz-2_r$1_s$2_${TAG}.json"
  done
  # A counter proves cuEquivariance ran; only a per-call timing at the run's own recorded shape says
  # what it cost. Third card on the same probe at the same shape, so all three are comparable.
  step probe_cueq 900 /root/venv-boltz/bin/python cueq_tri_probe.py \
    --from-json "$R/phase_boltz-2_r3_s200_${TAG}.json" --out "$R/cueq_probe_${TAG}.json"
  # OpenFold3 is the fp32 row and the one that compresses the A100-vs-H200 gap most (1.651x). TAG
  # ...sat so the file cannot be confused with the published openfold3 cell (16.942 s).
  step sat_of3 2400 env TAG="${TAG}sat" MODELS="openfold3" bash gpu5_session.sh
  # The cores ladder matters more here than on either other box: this is the campaign's only AMD
  # host (EPYC 7513, 30.72 vCPU) against two Intel Xeons, and that is the one confound the prior
  # A100 pass named and could not close.
  for n in 12 6; do
    step "cores$n" 2400 taskset -c "0-$((n - 1))" /root/venv-boltz/bin/python gpu5_bench.py \
      --model boltz-2 --repeat 3 --power \
      --yaml "$REPO/perf/size512/fixtures/cdk2x2_512.yaml" \
      --a3m "$REPO/perf/size512/fixtures/cdk2x2_512.a3m" --work /root/work \
      --out "$R/cores${n}_boltz-2_${TAG}.json"
  done

  # --- NEW MODEL: Nesso-1, last and droppable ---------------------------------------------------
  # Nesso-1 merged after the manifest froze and its only GPU number is an H200 NVL at a 600 W limit
  # (perf/nesso1/gpu_reference.json), a different part from the page's 700 W SXM column. So there is
  # no A100 number for it anywhere, and this box is the only one in the campaign that can produce
  # one. Run gpu_nesso1_run.py directly rather than gpu_nesso1_sweep.py: the sweep has no rung
  # filter and would run all ten ladder cells plus six ligand cells, sixteen where two are owed.
  # --reps 4 and --refine on are the reference's own settings for ladder_aa512_* (rep 0 cold and
  # discarded, warm n=3), so these two cells are protocol-matched to the H200 numbers they compare to.
  if [ "${MEASURE_NESSO:-0}" = 1 ] && [ -x /work/v_nesso/bin/python ]; then
    for k in cueq torch; do
      NK=""; [ "$k" = torch ] && NK="--no-kernels"
      step "nesso_aa512_$k" 1800 /work/v_nesso/bin/python "$W/perf/nesso1/gpu_nesso1_run.py" \
        --inputs "$W/perf/nesso1/inputs/ladder/aa512" \
        --out-dir "/work/out/ladder_aa512_$k" \
        --report "$R/nesso1_ladder_aa512_${k}_${TAG}.json" \
        --reps 4 --refine on --label "ladder_aa512_$k" $NK
    done
  else
    say "MEASURE_NESSO not set or /work/v_nesso missing: no Nesso-1 row on this box"
  fi

  date -u +%FT%TZ > "$R/MEASURE_DONE"
  say "MEASURE_DONE"
}

case "${1:-all}" in
  setup)   do_setup ;;
  measure) do_measure ;;
  all)     do_setup; do_measure ;;
  measure-h200) do_measure_h200 ;;
  all-h200)     SETUP_DESIGN=0 do_setup; do_measure_h200 ;;
  measure-a100) do_measure_a100 ;;
  # Nesso-1 landed after the freeze and this is the only box left that can give it a GPU column, so
  # both nesso knobs default ON here. Pass SETUP_NESSO=0 MEASURE_NESSO=0 if the rental-time
  # origin/main check comes back saying it is not there after all.
  all-a100)     SETUP_DESIGN=0
                SETUP_NESSO=${SETUP_NESSO:-1}
                MEASURE_NESSO=${MEASURE_NESSO:-1}
                export SETUP_DESIGN SETUP_NESSO MEASURE_NESSO
                do_setup; do_measure_a100 ;;
  *) echo "usage: $0 {setup|measure|all|measure-h200|all-h200|measure-a100|all-a100}" >&2; exit 2 ;;
esac

say "INVENTORY"
ls -l "$R"/*.json "$R"/*.jsonl 2>/dev/null | tee -a "$LOG"
ls "$R"/STEP_*.fail 2>/dev/null | tee -a "$LOG"
date -u +%FT%TZ > "$R/ALL_DONE"
say "ALL_DONE"
