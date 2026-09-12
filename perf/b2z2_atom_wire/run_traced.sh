#!/usr/bin/env bash
# The arm that decides this row: the same three folds with the diffusion loop TRACED.
#
# Untraced, the sharded step measured 0.99380x of the same mesh -- the shard's extra programs
# (two mesh_partition + two all_gather on q, plus one halo all_gather per atom layer, six per
# step) are all paid at host-dispatch rates, and on a mesh dispatch costs twice. Tracing removes
# exactly that cost, which is why b2z2-dual-chip-fold measured the trace at 1.0404x on a mesh and
# refuted it at 0.9948x on one chip: the same lever inverts with topology.
#
# Arm order is mesh, shard, mesh -- the second mesh is the A/A floor AND it brackets the shard in
# time, so box drift shows up as a floor wider than the effect rather than as the effect.
set -u
cd "$(dirname "$0")/../.."
ROOT=$PWD
PY=${PY:-/home/mthuening/work/tt-bio/env/bin/python3}
N=${N:-2}
CARDS=${CARDS:-16,17}
ONE=${ONE:-16}
REPS=${REPS:-5}
TAG=${TAG:-trace}
OUTDIR=${OUTDIR:-$ROOT/perf/b2z2_atom_wire}
ARMS=${ARMS:-"mesh:A shard:A mesh:B"}

run () {  # mode cards out
  echo "=== $1 on cards $2 -> $3 ($(date -u +%H:%M:%S)) ==="
  env TT_VISIBLE_DEVICES=$2 TT_BIO_LEASE_CARDS=$2 FOLD_MESH_N=$N FOLD_OUT=$3 \
      TT_BIO_LEASE_HOLDER=worker:b2z2-atom-shard-wire TT_BIO_TRACE_REGION_SIZE=536870912 \
      FOLD_REPS=$REPS FOLD_TRACE=${TRACE:-1} FOLD_SCRATCH=/tmp/b2z2_atomwire_$(id -u)_$4 \
      $PY "$ROOT/perf/b2z2_atom_wire/wire_fold.py" "$1" < /dev/null
  echo "=== $1 exit $? ($(date -u +%H:%M:%S)) ==="
}

for spec in $ARMS; do
  arm=${spec%%:*}; rep=${spec##*:}
  out="$OUTDIR/fold_${arm}_n${N}_${TAG}_${rep}.json"
  if [ "$arm" = "single" ]; then run single "$ONE" "$out" "$rep"; else run "$arm" "$CARDS" "$out" "$rep"; fi
done
echo ALLDONE
