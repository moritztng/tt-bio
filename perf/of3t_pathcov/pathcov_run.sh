#!/usr/bin/env bash
# of3t-pathcov: take the call-site census on one of the arms the model union is built from.
#
#   pathcov_run.sh diffusion [structs]   perf/of3t_diffusion/device_gradient.py, the shipped
#                                        diffusion arm behind MODEL_shipped.json
#   pathcov_run.sh cond                  perf/of3t_conditioning/device_gradient.py
#   pathcov_run.sh msa | aux             perf/of3t_auxheads/{msa,aux}_instrument.py
#   pathcov_run.sh trunk [blocks]        perf/of3t_trunkg043/dev_grad.py, the 48-block trunk.
#                                        Site census only: it is taped at a different boundary
#                                        than the union, so its mass does not compose.
#
# The instrument is run unmodified under perf/of3t_pathcov/census.py. Card 2 is this row's
# grant; TT_BIO_LEASE_CARDS pins the open so a four-chip bring-up cannot happen by accident.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
source "$W/perf/refpath.sh" 2>/dev/null || true
PY=/home/ttuser/tt-bio-dev/env/bin/python
CARD=${CARD:-2}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=${LEASE_CARDS:-$CARD}
export TT_BIO_LEASE_HOLDER=worker:of3t-pathcov
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
D=/tmp/of3t/of3t-pathcov
mkdir -p "$D"
ARM=${1:?arm}; shift
case "$ARM" in
  diffusion)
    export PYTHONPATH="/home/ttuser/of3t_refprec/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_tape:$W/perf/of3t_gradients:$W"
    SCRIPT=perf/of3t_diffusion/device_gradient.py
    ARGS=(--structs "${1:-all}" --tag _pathcov --cap /home/ttuser/of3t_softgrad/diffcap043
          --out-dir "$D") ;;
  cond)
    export PYTHONPATH="$W/perf/of3t_tape:$W"
    SCRIPT=perf/of3t_conditioning/device_cond_gradient.py
    ARGS=(--tag _pathcov --out-dir "$D") ;;
  msa|aux)
    export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" "$W/perf/of3t_tape" "$W")"
    REF=$REF_BUNDLE/grads_f64_043.pt
    if [ "$ARM" = aux ]; then
      SCRIPT=perf/of3t_auxheads/aux_instrument.py
      BND=/home/ttuser/of3t_auxheads/cap043/boundary_aux_heads.pt
    else
      SCRIPT=perf/of3t_auxheads/msa_instrument.py
      BND=/home/ttuser/of3t_auxheads/cap043b/boundary_msa_module.pt
    fi
    ARGS=(--boundary "$BND" --reference-grads "$REF" --out "$D/instrument_${ARM}_pathcov.json") ;;
  trunk)
    # NOT one of the four the model union is built from -- `model_scope.py` refuses it entry
    # because it is taped at crop64/block47 and the union is fixed at batch_step003. It is
    # censused anyway because the SITE table needs no reference names: it answers whether the
    # 48-block trunk instance executes the same call sites the aux arm's embedded 4-block
    # stack does, which is what separates "this branch never runs" from "no arm in the union
    # taped this instance of it".
    export PYTHONPATH="$W/perf/of3t_gradients:$W/perf/of3t_tape:$W"
    SCRIPT=perf/of3t_trunkg043/dev_grad.py
    ARGS=(--boundary /home/ttuser/of3t_trunk043ref/boundary_c64.pt
          --cap-last /home/ttuser/of3t_gradients/cap/block47_boundary.pt --arm flipped
          --blocks "${1:-48}"
          --out "$D/dev_trunk_pathcov.pt"
          --report "$D/DEV_trunk_pathcov.json") ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
S=$(date +%s)
echo "=== pathcov census, arm $ARM, card $CARD, $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_pathcov/census.py --out "perf/of3t_pathcov/CENSUS_${ARM}.json.gz" \
      -- "$SCRIPT" "${ARGS[@]}"
rc=$?
echo "=== pathcov census $ARM exit $rc elapsed $(( $(date +%s) - S ))s ==="
exit $rc
