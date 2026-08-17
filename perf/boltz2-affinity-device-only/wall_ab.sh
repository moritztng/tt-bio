#!/usr/bin/env bash
# wall_ab.sh <tree> <tag> <reps>
# FKBP12+SB3 at the shipped affinity protocol. Two arms: with affinity property
# (affinity arm) and without (structure arm). Warm: rep 1 is discarded by the caller.
set -u
TREE="$1"; TAG="$2"; REPS="${3:-4}"
OUT=/home/ttuser/.coworker/wt/boltz2-affinity-device-only/perf/boltz2-affinity-device-only
YAML_S=$OUT/fkg_structure_only.yaml
YAML_A=$TREE/examples/affinity_fkg.yaml
cd "$TREE"
export PYTHONPATH="$TREE" TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:boltz2-affinity-device-only
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FLAGS_COMMON="--model boltz2 --single_sequence --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 --seed 0"
for arm in structure affinity; do
  if [ "$arm" = structure ]; then Y=$YAML_S; EXTRA=""; else Y=$YAML_A; EXTRA="--affinity_mw_correction --sampling_steps_affinity 200 --diffusion_samples_affinity 5"; fi
  for r in $(seq 1 $REPS); do
    D=$(mktemp -d /tmp/wall_${TAG}_${arm}_XXXX)
    S=$(date +%s.%N)
    $PY -m tt_bio.main predict "$Y" --out_dir "$D" $FLAGS_COMMON $EXTRA > "$D/run.log" 2>&1
    RC=$?
    E=$(date +%s.%N)
    echo "$TAG $arm rep$r rc=$RC wall=$(echo "$E-$S"|bc)" | tee -a $OUT/wall_$TAG.txt
    [ $RC -ne 0 ] && tail -20 "$D/run.log" >> $OUT/wall_$TAG.txt
    rm -rf "$D"
  done
done
