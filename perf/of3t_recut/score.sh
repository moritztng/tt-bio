#!/usr/bin/env bash
# of3t-recut job 4: the graded model artifact with the CORRECTED trunk arm, and the clause read
# against of3t-modelframe's pre-registered ladder. No new bar is written here.
#
#   score.sh aa     the A/A control: the BANKED frame-matched trunk arm through this row's own
#                   invocation, which must reproduce 0.27095922968432157 exactly. Until it does,
#                   nothing below it may be read.
#   score.sh arm    the corrected trunk arm, then clause.py.
#
# This is `perf/of3t_modelframe/score.sh` with one --arm path changed and a different --out. The
# five other arms, the three pinned reference digests, the section table and the unwrap key are
# its own. CPU only, no card is opened.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-recut
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
O=/home/ttuser/of3t_recut
COV=/home/ttuser/of3t_covadopt
mkdir -p "$O"

MODE=${1:-arm}
case "$MODE" in
  aa)  TRUNK=/home/ttuser/of3t_modelframe/dev_RENORM_model_n384_nocaptures.pt
       OUT=perf/of3t_recut/AA_FRAMEMATCHED_composed3660_n384.json; SIDE=sidecar_aa ;;
  arm) TRUNK=${2:-$O/dev_RENORM_model_n384_corrected.pt}
       OUT=perf/of3t_recut/MODEL_RECUT_composed3660_n384.json; SIDE=sidecar_recut ;;
  *) echo "usage: score.sh [aa|arm [trunk.pt]]"; exit 2 ;;
esac
[ -s "$TRUNK" ] || { echo "missing trunk arm $TRUNK"; exit 2; }

OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_wholemodel/model_scope.py \
  --arm "renorm:diffusion=$COV/device_grads_rc_refatom_on.pt" \
  --arm "renorm:input_embedder=$COV/ie_grads_f64_renorm_on.pt" \
  --arm "renorm:cond=/home/ttuser/of3t_wholemodel/cond_grads_renorm.pt" \
  --arm "renorm:aux=/home/ttuser/of3t_wholemodel/aux_grads_renorm.pt" \
  --arm "renorm:msa=/home/ttuser/of3t_wholemodel/msa_grads_renorm.pt" \
  --arm "renorm:pairformer_stack=$TRUNK" \
  --unwrap-key grads \
  --f64  /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt \
  --bf16 /home/ttuser/of3t_refprec/pinned_p175/arm4_bf16_autocast/grads_f64.pt \
  --f32  /home/ttuser/of3t_refprec/pinned_p175/arm2_f32_upstream/grads_f64.pt \
  --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
  --expect float64=1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4 \
  --expect upstream_bf16=ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb \
  --expect upstream_f32=09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548 \
  --out "$OUT" --sidecar-dir "$O/$SIDE" 2>&1 | tail -25
echo "=== model_scope exit ${PIPESTATUS[0]} mode $MODE out $OUT ==="

if [ "$MODE" = aa ]; then
  "$PY" - "$OUT" <<'PY'
import json, sys
PUB = 0.27095922968432157
d = json.load(open(sys.argv[1]))
got = d["stats"]["renorm_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"]
rel = abs(got - PUB) / PUB
print(json.dumps({"what": "A/A: of3t-modelframe's frame-matched headline through this row's "
                          "own invocation of the same scorer",
                  "published_headline": PUB, "reproduced": got, "rel_difference": rel,
                  "bit_identical": got == PUB,
                  "verdict": "same instrument, same inputs" if rel == 0.0 else
                             "this row's invocation is NOT the published one"}, indent=1))
sys.exit(0 if rel == 0.0 else 1)
PY
  echo "=== AA exit $? ==="
else
  "$PY" perf/of3t_modelframe/clause.py \
    --published perf/of3t_modelboundary/MODEL_withtrunk_composed3660_n384.json \
    --rescored  "$OUT" \
    --out       perf/of3t_recut/CLAUSE_RECUT.json
fi
