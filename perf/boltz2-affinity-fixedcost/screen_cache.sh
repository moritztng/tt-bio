#!/bin/bash
# Screen: what does the unconditional program-cache clear in Boltz2.forward actually cost?
#
# Two identical-shape targets in one invocation, so target 2 is the maximum-possible win: with the
# clear suppressed its whole cache (structure 253 + affinity 298 entries) is already warm. Target 1
# is cold in both arms and doubles as the A/A control.
set -u
WT=/home/ttuser/.coworker/wt/boltz2-affinity-program-cache-fixed-cost
cd "$WT" || exit 1
D=perf/boltz2-affinity-fixedcost
P=/home/ttuser/tt-bio-dev/env/bin/python3
AA=${AA:-256}
ARM=${ARM:-A}
KEEP=0
[ "$ARM" = "B" ] && KEEP=1

IN="$D/work/screen_in"
mkdir -p "$IN"
"$P" - "$AA" "$IN" <<'PY'
import sys, pathlib
sys.path.insert(0, "perf/nesso1")
from make_inputs import LADDER_LIGAND, cdk2, yaml_for
aa = int(sys.argv[1]); d = pathlib.Path(sys.argv[2])
y = yaml_for(cdk2(aa), LADDER_LIGAND)
for i in (1, 2):
    (d / ("t%d.yaml" % i)).write_text(y)
PY

OUT="$D/work/screen_out_${ARM}"
rm -rf "$OUT"
RUN="$D/results/screen_${ARM}.log"
S=$(date +%s.%N)
env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-affinity-program-cache-fixed-cost \
    PYTHONPATH="$WT" TT_BIO_BOLTZ2_KEEP_PROGRAM_CACHE="$KEEP" \
    "$P" -u -m tt_bio.main predict "$IN" --out_dir "$OUT" \
    --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
    --sampling_steps_affinity 200 --diffusion_samples_affinity 5 \
    --affinity_mw_correction --output_format cif --single_sequence --override \
    --accelerator tenstorrent --model boltz2 --num_devices 1 > "$RUN" 2>&1
RC=$?
E=$(date +%s.%N)
echo "arm=$ARM keep=$KEEP rc=$RC wall_s=$(echo "$E - $S" | bc)" | tee -a "$D/results/screen_walls.txt"
grep -E "✓" "$RUN" | tee -a "$D/results/screen_walls.txt"
"$P" -c "
import json, glob, sys
for f in sorted(glob.glob('$OUT/*/results.json')):
    for r in json.load(open(f)):
        print(r['id'], 'runtime_s', r.get('runtime_s'), 'struct', r.get('structure_runtime_s'),
              'aff', r.get('affinity_runtime_s'), 'pred', r.get('affinity_pred_value'))
" | tee -a "$D/results/screen_walls.txt"
