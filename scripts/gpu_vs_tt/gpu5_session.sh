#!/usr/bin/env bash
# One command per rented GPU: run the five 512 aa cells back to back, gate each one, leave a
# JSON per cell in /root/results. Detached-safe, so it survives the launching shell.
#
#   bash gpu5_session.sh                    # all five, in the task's order
#   MODELS="boltz-2 esmfold2" bash gpu5_session.sh
#
# Not -e: one model failing to run must leave the other four measured. A cell that dies
# writes its traceback to its own log and the table records the reason.
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
R=/root/results
mkdir -p "$R"
TAG=${TAG:-h200}
MODELS=${MODELS:-"protenix-v2 opendde boltz-2 esmfold2 openfold3"}
REPEAT=${REPEAT:-3}
BUDGET_S=${BUDGET_S:-5400}
START=$(date +%s)
export CUDA_HOME=/opt/conda
export PATH=/opt/conda/bin:$PATH
FIX=$HERE/../../perf/size512/fixtures

gate() {  # gate <structure> <label>
  [ -s "$1" ] || { echo "GATE: no structure at $1"; return 1; }
  python3 "$HERE/gpu5_accuracy_gate.py" "$1" --expect-residues 512 ${2:+--expect-plddt "$2"}
}

for M in $MODELS; do
  EL=$(( $(date +%s) - START ))
  if [ "$EL" -gt "$BUDGET_S" ]; then
    echo "== SKIP $M: ${EL}s exceeds BUDGET_S=$BUDGET_S =="; continue
  fi
  echo "== $M on $TAG: $(date -u +%FT%TZ) (${EL}s in) =="
  OUT=$R/gpu_${M}_prot512_${TAG}.json
  LOG=$R/${M}_${TAG}.log
  ST=$R/struct_${M}_${TAG}
  mkdir -p "$ST"
  case "$M" in
    protenix-v2|opendde)
      # gpu_bench.py drives the protenix family: model loaded once, runner.predict timed
      # with a CUDA sync on both sides, cold fold reported separately.
      if [ "$M" = "protenix-v2" ]; then
        PY=/root/venv-protenix/bin/python3; CK=/root/ckpt/protenix-v2.pt; RUNG=LD-shipped-default
      else
        PY=/root/venv-opendde/bin/python3; CK=/root/ckpt/opendde.pt; RUNG=L2-bf16-fusion-cache
      fi
      $PY gpu_bench.py --model "$M" --repeat "$REPEAT" --checkpoint "$CK" \
        --msa-a3m fixtures/prot512.a3m --seq-file fixtures/prot512.seq \
        --label "cdk2x2_512 (512 aa)" --name prot512 --rungs "$RUNG" \
        --save-structure "$ST" --out "$OUT" > "$LOG" 2>&1
      echo "$M rc=$?"
      # protenix-v2 reads plDDT 0.828628 on this fixture on the TT side, so the gate
      # prints a delta against a value already on record for the same model and fixture.
      [ "$M" = "protenix-v2" ] && EXP=0.828628 || EXP=
      gate "$ST/$RUNG.pdb" "$EXP" | tee "$R/gate_${M}_${TAG}.txt"
      ;;
    boltz-2|openfold3|esmfold2)
      case "$M" in
        boltz-2)   PY=/root/venv-boltz/bin/python3 ;;
        openfold3) PY=/root/venv-of3/bin/python3 ;;
        esmfold2)  PY=/root/venv-esm/bin/python3 ;;
      esac
      $PY gpu5_bench.py --model "$M" --repeat "$REPEAT" \
        --yaml "$FIX/cdk2x2_512.yaml" --a3m "$FIX/cdk2x2_512.a3m" \
        --seq-file fixtures/prot512.seq --work /root/work \
        --checkpoint /root/ckpt/of3-p2-155k.pt --out "$OUT" > "$LOG" 2>&1
      echo "$M rc=$?"
      # gpu5_bench.py records every prediction it produced; gate the last one, which is
      # the last warm fold rather than the discarded cold one.
      P=$(python3 -c "
import json,sys
try:
    p=json.load(open('$OUT'))['result'].get('predictions') or []
    print(p[-1] if p else '')
except Exception: print('')
")
      gate "$P" "" | tee "$R/gate_${M}_${TAG}.txt"
      ;;
  esac
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,power.draw --format=csv,noheader \
    >> "$R/nvidia_${TAG}.txt"
done

echo "== session end $TAG: $(date -u +%FT%TZ) total=$(( $(date +%s) - START ))s =="
ls -l "$R"/*.json
