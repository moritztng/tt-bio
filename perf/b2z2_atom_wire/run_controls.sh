#!/usr/bin/env bash
# Bit-exactness at every mesh width, with the two controls that make the check mean something.
#
# Per width: the shard must write the one-chip fold's digest, a shard with the per-layer halo
# exchange DROPPED must not, and a shard whose slab is scaled by 1 + 2**-8 must not. The first
# alone proves nothing -- a gate that declined, a break that never fired and a shard that worked
# all write the same bytes -- so each arm records the gate counter and the break counter and the
# harness asserts both against the arm it was asked for.
#
# One fold per arm (FOLD_REPS=0 keeps the cold fold): a digest does not care that the fold which
# produced it also compiled the kernels, and this runs while the box is too contended to time.
set -u
cd "$(dirname "$0")/../.."
ROOT=$PWD
PY=${PY:-/home/mthuening/work/tt-bio/env/bin/python3}
CARDS=${CARDS:-20,21,22,23}
WIDTHS=${WIDTHS:-"2 4"}
OUTDIR=${OUTDIR:-$ROOT/perf/b2z2_atom_wire}

for n in $WIDTHS; do
  cards=$(echo "$CARDS" | cut -d, -f1-$n)
  for brk in "" halo perturb; do
    tag=${brk:-clean}
    out="$OUTDIR/ctl_shard_n${n}_${tag}.json"
    echo "=== n=$n break=${brk:-none} on $cards -> $out ($(date -u +%H:%M:%S)) ==="
    env TT_VISIBLE_DEVICES=$cards TT_BIO_LEASE_CARDS=$cards FOLD_MESH_N=$n FOLD_OUT=$out \
        TT_BIO_LEASE_HOLDER=worker:b2z2-atom-shard-wire TT_BIO_TRACE_REGION_SIZE=536870912 \
        FOLD_REPS=0 FOLD_TRACE=0 FOLD_BREAK="$brk" \
        FOLD_SCRATCH=/tmp/b2z2_atomwire_$(id -u)_ctl${n}${tag} \
        $PY "$ROOT/perf/b2z2_atom_wire/wire_fold.py" shard < /dev/null
    echo "=== n=$n break=${brk:-none} exit $? ($(date -u +%H:%M:%S)) ==="
  done
done
echo ALLDONE
