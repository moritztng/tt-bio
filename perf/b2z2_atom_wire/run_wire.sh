#!/usr/bin/env bash
# Three arms of the same fold, one after the other in the same box state, each in its own process
# because the device open differs. Single is the control and the fold ratio's denominator; mesh
# isolates the mesh tax; shard is the lever. All three must write the same CIF.
set -u
cd "$(dirname "$0")/../.."
ROOT=$PWD
PY=${PY:-/home/mthuening/work/tt-bio/env/bin/python3}
N=${N:-2}
CARDS=${CARDS:-12,13}
ONE=${ONE:-12}
REPS=${REPS:-3}
TRACE=${TRACE:-0}          # trace the diffusion loop; on a mesh this is where the shard has to pay
ARMS=${ARMS:-"single mesh shard"}
OUTDIR=${OUTDIR:-$ROOT/perf/b2z2_atom_wire}
COMMON="TT_BIO_LEASE_HOLDER=worker:b2z2-atom-shard-wire TT_BIO_TRACE_REGION_SIZE=536870912 FOLD_REPS=$REPS"

run () {  # mode cards out
  echo "=== $1 on cards $2 -> $3 ==="
  env TT_VISIBLE_DEVICES=$2 TT_BIO_LEASE_CARDS=$2 FOLD_MESH_N=$N FOLD_OUT=$3 \
      TT_BIO_LEASE_HOLDER=worker:b2z2-atom-shard-wire TT_BIO_TRACE_REGION_SIZE=536870912 \
      FOLD_REPS=$REPS FOLD_TRACE=$TRACE $PY "$ROOT/perf/b2z2_atom_wire/wire_fold.py" "$1"
  echo "=== $1 exit $? ==="
}

SUF=""; [ "$TRACE" = "1" ] && SUF="_trace"
for arm in $ARMS; do
  case $arm in
    single) run single "$ONE"   "$OUTDIR/fold_single_n1$SUF.json" ;;
    mesh)   run mesh   "$CARDS" "$OUTDIR/fold_mesh_n$N$SUF.json" ;;
    shard)  run shard  "$CARDS" "$OUTDIR/fold_shard_n$N$SUF.json" ;;
  esac
done
echo ALLDONE
