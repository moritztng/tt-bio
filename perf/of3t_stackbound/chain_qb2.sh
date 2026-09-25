#!/usr/bin/env bash
# of3t-stackbound on qb2 (relocated from qb1, orchestrator pass 418). pc cannot hold the
# capture's 44 GB peak, so the CPU half runs here too, and never beside one of this row's arms:
#   f64 ref -> (capture || f32 ref) -> cc -> COTANGENT_COMPLETE -> six arms on card 3 (p300c).
# Each step is skipped if its output already exists, so a relaunch resumes where it stopped.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-stackbound
H=$W/perf/of3t_stackbound
O=/home/ttuser/of3t_stackbound
L=$O/chain_qb2.log
cd "$W"
log() { echo "=== $* $(date -u +%FT%TZ) ===" | tee -a "$L"; }
step() { local tag=$1; shift; "$@" > "$O/$tag.log" 2>&1; local rc=$?; log "$tag rc $rc"; return $rc; }

[ -s "$O/ref_f64/grads_f64.pt" ] || step ref_f64 bash "$H/refs.sh" f64 || exit 1
if [ ! -s "$O/cot_model_sb.pt" ] || [ ! -s "$O/ref_f32/grads_f64.pt" ]; then
  [ -s "$O/cot_model_sb.pt" ] || { step capture bash "$H/frame.sh" capture & }
  [ -s "$O/ref_f32/grads_f64.pt" ] || { step ref_f32 bash "$H/refs.sh" f32 & }
  wait
fi
[ -s "$O/cot_model_sb.pt" ] && [ -s "$O/ref_f32/grads_f64.pt" ] || { log "capture or f32 missing"; exit 1; }
[ -s "$O/ref_f64_trunk_sb_corrected.pt" ] || step cc bash "$H/frame.sh" cc || exit 1
[ -s "$O/cot_external_sb.pt" ] || step frame_check /home/ttuser/tt-bio-dev/env/bin/python "$H/frame_check.py" || exit 1
python3 -c "import json,sys;sys.exit(0 if json.load(open('$H/COTANGENT_COMPLETE.json'))['COTANGENT_COMPLETE'] else 1)" \
  || { log "COTANGENT_COMPLETE FAILED -- no arm runs (A40)"; exit 2; }
export ARM_CARD=3 ARM_BOARD=p300c
for a in "SHIP_A none" "SL softmax,layer_norm" "SHIP_B none" "SLZ softmax,layer_norm z_fp32_residual=1" \
         "S softmax" "L layer_norm"; do
  set -- $a
  [ -s "$O/dev_$1.pt" ] || step "dev_$1" bash "$H/arm.sh" "$@"
done
log "CHAIN DONE"
