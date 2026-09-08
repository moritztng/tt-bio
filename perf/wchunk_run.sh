#!/bin/bash
# One size-ladder-config fold. $1=label $2=card $3=rung $4=model
#   env WTHR sets the W-chunk threshold override (unset = whatever the source tree decides)
#   env SRC  picks the tt_bio source tree (unset = this worktree). Point it at
#            perf/wchunk/base_a3186e41 to run the pre-fix code as an A/B arm.
set -u
WT=/home/moritz/.coworker/wt/wh-transition-wchunk-hang-fix
LABEL=$1; CARD=$2; RUNG=$3; MODEL=${4:-protenix-v2}
SRC=${SRC:-$WT}
mkdir -p $WT/perf/wchunk
# cd to the SOURCE tree, not the worktree: for both -c and -m, python puts the cwd at
# sys.path[0], AHEAD of PYTHONPATH. Running from $WT with PYTHONPATH=$SRC therefore imported
# the worktree anyway and made the two A/B arms the same code -- caught 2026-09-07 by the
# resolved-path echo below, which is why that echo is now a hard assertion.
cd $SRC
export PYTHONNOUSERSITE=1 PYTHONPATH=$SRC
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:wh-transition-wchunk-hang-fix
export TT_BIO_SIZE_LIMIT=0
[ -n "${WTHR:-}" ] && export TT_BIO_TRANSITION_W_CHUNKING_THRESHOLD=$WTHR
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
echo "=== $LABEL card=$CARD rung=$RUNG model=$MODEL WTHR=${WTHR:-shipped} src=$SRC start $(date -u +%FT%TZ)"
# Which tt_bio actually got imported. A cwd or .pth shadow that beats PYTHONPATH would
# silently make both A/B arms the same code, so refuse to run rather than report a fake match.
RESOLVED=$($PY -c "import tt_bio; print(tt_bio.__file__)")
echo "=== tt_bio resolved: $RESOLVED"
case "$RESOLVED" in
  "$SRC"/tt_bio/*) ;;
  *) echo "=== $LABEL ABORT: tt_bio resolved outside SRC=$SRC -- arm would be vacuous"; exit 3 ;;
esac
$PY -u -m tt_bio.main predict \
    $WT/perf/size512/fixtures/cdk2x2_${RUNG}.yaml \
    --model $MODEL --single_sequence --sampling_steps 6 --diffusion_samples 1 --seed 0 \
    --out_dir $WT/perf/wchunk/out_$LABEL
rc=$?
# Structure digests, so two arms can be compared bit-exact without a second pass.
find $WT/perf/wchunk/out_$LABEL -name "*.cif" -o -name "*.pdb" | sort | xargs -r md5sum
echo "=== $LABEL rc=$rc end $(date -u +%FT%TZ)"
