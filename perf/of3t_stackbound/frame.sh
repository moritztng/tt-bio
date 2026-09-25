#!/usr/bin/env bash
# of3t-stackbound: the 4hhb model frame, CPU only on qb1, no card opened.
#
#   frame.sh capture   capture_model_frame.py unmodified on the 4hhb batch: the trunk boundary
#                      and the cotangent flowing back into it, from a float64 full-model
#                      backward that must reproduce ref_f64's loss bit for bit (the witness).
#   frame.sh cc        ref_grad.py --policy f64 (graph-cut-external, D242 repaired) driven by
#                      that capture. Its trunk gradient against ref_f64's pairformer_stack
#                      section is COTANGENT_COMPLETE (frame_check.py), and its cot_z_correction
#                      is the one correction every device arm is driven by (A42).
#
# perf/of3t_modelframe/capture.sh and perf/of3t_recut/n384_arms.sh with the batch, draws,
# reference and out paths changed and nothing else: same tree, crop, threads, producers.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-stackbound
cd "$W"
source perf/refpath.sh
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" /home/ttuser/of3t_frame384/deps)"
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
O=/home/ttuser/of3t_stackbound
B_SHA=667b7530e421eba525fbb630d925d182e4e5abd5c1d357cfaea058b9b03e54b3
TH=14
export OMP_NUM_THREADS=$TH
S=$(date +%s)
case "${1:?usage: frame.sh capture|cc}" in
  capture)
    LOSS=$("$PY" -c "import json;print(repr(json.load(open(\"$O/ref_f64/manifest.json\"))[\"loss\"]))")
    GN=$("$PY" -c "import json;d=json.load(open(\"$O/ref_f64/manifest.json\"));print(repr(d[\"gradient\"][\"global_norm\"]))")
    echo "=== capture start $(date -u +%FT%TZ) host $(hostname) threads $TH expect loss $LOSS norm $GN ==="
    nice -n 5 "$PY" perf/of3t_modelframe/capture_model_frame.py \
      --batch "$O/batch_step002.pt" --batch-sha256 $B_SHA \
      --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
      --replay-draws "$O/ref_f64/draws.pt" --ref-grads "$O/ref_f64/grads_f64.pt" \
      --expect-loss "$LOSS" --expect-grad-norm "$GN" --seed 20260923 \
      --out-dir "$O" --threads $TH --tag model_sb 2>&1 \
      | grep -vE "UserWarning|warnings.warn|^  from openfold3|Consider using tensor.detach"
    rc=${PIPESTATUS[0]} ;;
  cc)
    echo "=== cc start $(date -u +%FT%TZ) host $(hostname) threads $TH ==="
    nice -n 5 "$PY" perf/of3t_trunkg043/ref_grad.py --tree "$REF_OF3PKG" \
      --boundary "$O/boundary_model_sb.pt" --cap-last "$O/cot_model_sb.pt" \
      --policy f64 --blocks 48 --crop 384 --threads $TH --checkpoint \
      --out "$O/ref_f64_trunk_sb_corrected.pt" \
      --report perf/of3t_stackbound/REF_F64_TRUNK_SB_CORRECTED.json 2>&1 \
      | grep -vE "UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach|loss\)\}, a.out\)"
    rc=${PIPESTATUS[0]} ;;
  *) echo "unknown step $1"; exit 2 ;;
esac
echo "=== $1 exit $rc elapsed $(($(date +%s)-S))s $(date -u +%FT%TZ) ==="
exit $rc
