#!/bin/bash
# Phase-1 attribution trace: one real Boltz-2 affinity prediction, fully instrumented.
#
# Fixture and CLI args are the nesso1-perf-p2 protocol verbatim (CDK2 tiled to N aa x the
# upstream README ligand, 22 heavy atoms, no MSA, tt-bio's shipped affinity settings), so the
# breakdown below attributes the same 326.5 s invocation that pass measured.
set -u
WT=/home/ttuser/.coworker/wt/boltz2-affinity-program-cache-fixed-cost
cd "$WT" || exit 1
D=perf/boltz2-affinity-fixedcost
AA=${AA:-256}
TAG=${TAG:-base}
STEPS_AFF=${STEPS_AFF:-200}
SAMPLES_AFF=${SAMPLES_AFF:-5}
P=/home/ttuser/tt-bio-dev/env/bin/python3

mkdir -p "$D/results" "$D/work"
IN="$D/work/in_${AA}"
mkdir -p "$IN"
"$P" - "$AA" "$IN" <<'PY'
import sys, pathlib
sys.path.insert(0, "perf/nesso1")
from make_inputs import LADDER_LIGAND, cdk2, yaml_for
aa = int(sys.argv[1]); d = pathlib.Path(sys.argv[2])
(d / ("cdk2_%d.yaml" % aa)).write_text(yaml_for(cdk2(aa), LADDER_LIGAND))
print("wrote", d)
PY

LOG="$D/results/trace_${TAG}_aa${AA}.jsonl"
RUN="$D/results/trace_${TAG}_aa${AA}.log"
rm -f "$LOG"
OUT="$D/work/out_${TAG}_${AA}"
rm -rf "$OUT"

S=$(date +%s.%N)
env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-affinity-program-cache-fixed-cost \
    PYTHONPATH="$WT:$WT/$D/probe" AFFPROBE_LOG="$WT/$LOG" \
    ${EXTRA_ENV:-} \
    "$P" -u -m tt_bio.main predict "$IN" --out_dir "$OUT" \
    --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
    --sampling_steps_affinity "$STEPS_AFF" --diffusion_samples_affinity "$SAMPLES_AFF" \
    --affinity_mw_correction --output_format cif --single_sequence --override \
    --accelerator tenstorrent --model boltz2 > "$RUN" 2>&1
RC=$?
E=$(date +%s.%N)
echo "rc=$RC wall_s=$(echo "$E - $S" | bc)" | tee -a "$RUN"
echo "wall_s=$(echo "$E - $S" | bc) rc=$RC tag=$TAG aa=$AA steps_aff=$STEPS_AFF samples_aff=$SAMPLES_AFF" \
    >> "$D/results/walls.txt"
grep -h affinity_pred_value -r "$OUT" 2>/dev/null | head -3
