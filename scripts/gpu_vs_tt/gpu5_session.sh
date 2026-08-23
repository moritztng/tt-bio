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
POWER=${POWER:-1}      # POWER= (empty) drops the 200 ms nvidia-smi sampler
BUDGET_S=${BUDGET_S:-5400}
PER_MODEL_S=${PER_MODEL_S:-1800}   # a hung cell must not eat the rest of the rental

# torch's nvrtc JIT path -- the jiterator ops, of which ESMFold2's structure module hits erfinv --
# dlopens libnvrtc-builtins.so.<major>.<minor> by bare name. In a uv/pip venv that library ships
# inside the nvidia-* wheels and is not on the loader path, so on the B200 the first jiterated op
# died with "nvrtc: error: failed to open libnvrtc-builtins.so.13.0" AFTER the 25 GB weight load,
# which reads as a Blackwell incompatibility rather than a missing search path. Prepending the
# wheels' own lib dirs fixes it and changes no math. Scoped to the esm312 rows on purpose: putting
# these dirs ahead of the loader path for boltz-2 / openfold3 could shift which cublas or cudnn
# they load, and those two rows exist to reproduce a published cell exactly.
nvidia_lib_path() {
  "$1" - <<'PY'
import glob, os, sys
dirs = []
for base in (p for p in sys.path if p.endswith("site-packages")):
    dirs += sorted(glob.glob(os.path.join(base, "nvidia", "*", "lib")))
print(":".join(dirs))
PY
}
START=$(date +%s)
FAILED=0   # rows that produced no usable number; the session exits non-zero if any

export CUDA_HOME=/opt/conda
export PATH=/opt/conda/bin:$PATH
FIX=$HERE/../../perf/size512/fixtures

gate() {  # gate <structure> <label>
  [ -s "$1" ] || { echo "GATE: no structure at $1"; return 1; }
  python3 "$HERE/gpu5_accuracy_gate.py" "$1" --expect-residues 512 ${2:+--expect-plddt "$2"}
}

# The box must be quiet before a row is timed, but "quiet" has to be measured on something we
# own. /proc/loadavg is not namespaced: on the 192-core shared host behind the B200 rental it
# read 15-31 while our own processes drew 0.0% CPU, so a loadavg gate would have SKIPPED every
# row and published nothing. Our own cgroup CPU draw is the quantity that actually perturbs a
# row, and a foreign process on the card is the other disqualifier.
MAXCORES=${MAXCORES:-2.0}
own_cores() {
  local a b
  a=$(awk '/usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)
  sleep 5
  b=$(awk '/usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)
  awk -v a="$a" -v b="$b" 'BEGIN{printf "%.2f", (b-a)/5000000.0}'
}
foreign_gpu() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true; }

for M in $MODELS; do
  for _ in $(seq 1 60); do
    CORES=$(own_cores); FG=$(foreign_gpu)
    awk -v c="$CORES" -v m="$MAXCORES" 'BEGIN{exit !(c<m)}' && [ "${FG:-0}" -eq 0 ] && break
    echo "== waiting before $M: own draw $CORES cores (max $MAXCORES), foreign GPU procs $FG =="; sleep 25
  done
  awk -v c="$CORES" -v m="$MAXCORES" 'BEGIN{exit !(c<m)}' && [ "${FG:-0}" -eq 0 ] || {
    echo "== SKIP $M: own draw $CORES cores / foreign GPU procs $FG still bad after 30 min =="; continue; }
  echo "== quiet check before $M: own draw $CORES cores, foreign GPU procs $FG, host loadavg $(cut -d' ' -f1 /proc/loadavg) (not namespaced) =="
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
      timeout "$PER_MODEL_S" $PY gpu_bench.py --model "$M" --repeat "$REPEAT" --checkpoint "$CK" \
        --msa-a3m fixtures/prot512.a3m --seq-file fixtures/prot512.seq \
        --label "cdk2x2_512 (512 aa)" --name prot512 --rungs "$RUNG" \
        --save-structure "$ST" --out "$OUT" > "$LOG" 2>&1
      echo "$M rc=$?"
      # protenix-v2 reads plDDT 0.828628 on this fixture on the TT side, so the gate
      # prints a delta against a value already on record for the same model and fixture.
      [ "$M" = "protenix-v2" ] && EXP=0.828628 || EXP=
      gate "$ST/$RUNG.pdb" "$EXP" | tee "$R/gate_${M}_${TAG}.txt"
      ;;
    boltz-2|openfold3|esmfold2|esmfold2-fast)
      EXTRA=""; NVLIB=""
      case "$M" in
        boltz-2)   PY=/root/venv-boltz/bin/python3 ;;
        openfold3) PY=/root/venv-of3/bin/python3 ;;
        # venv-esm312, not venv-esm: upstream esm needs python >=3.12 and the model class
        # comes from transformers. --esm-backend selects the fast kernel path, which is
        # NOT the default, and 10 loops / 100 requested steps is the paper's protocol and
        # what the TT arm runs.
        # ESM_VENV exists for one caller: the pinned harness-control arm, which re-runs this
        # exact row against the packages the published cell names (venv-esm312ctl) to tell
        # package drift apart from a box or harness problem. Default is unchanged.
        esmfold2)  PY=${ESM_VENV:-/root/venv-esm312}/bin/python
                   EXTRA="--esm-backend cuequivariance --recycles 10 --steps 100"
                   NVLIB="$(nvidia_lib_path "$PY")" ;;
        # Same class, same venv, same backend, one different checkpoint: --esm-repo resolves
        # from ESM_REPOS, so it is deliberately not passed here.
        esmfold2-fast) PY=${ESM_VENV:-/root/venv-esm312}/bin/python
                   EXTRA="--esm-backend cuequivariance --recycles 10 --steps 100"
                   NVLIB="$(nvidia_lib_path "$PY")" ;;
      esac
      timeout "$PER_MODEL_S" \
        ${NVLIB:+env LD_LIBRARY_PATH="$NVLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"} \
        $PY gpu5_bench.py --model "$M" --repeat "$REPEAT" ${POWER:+--power} \
        --yaml "$FIX/cdk2x2_512.yaml" --a3m "$FIX/cdk2x2_512.a3m" \
        --seq-file fixtures/prot512.seq --work /root/work \
        --checkpoint /root/ckpt/of3-p2-155k.pt $EXTRA --out "$OUT" > "$LOG" 2>&1
      echo "$M rc=$?"
      # gpu5_bench.py records every prediction it produced; gate the last one, which is
      # the last warm fold rather than the discarded cold one. Written to a file rather
      # than inlined: nesting python in a double-quoted $( ) inside this heredoc-generated
      # script ate the quotes once already and gated the empty string.
      P=$(OUTJSON="$OUT" python3 last_prediction.py)
      gate "$P" "" | tee "$R/gate_${M}_${TAG}.txt"
      ;;
  esac
  # A row that wrote a result JSON with a populated "error" is a FAILED row, not a measured one.
  # The first B200 pass marked seven such rows ok because the outer step gate keyed on the file
  # existing rather than on what was inside it, so esmfold2-fast's nvrtc failure was recorded as
  # "OK new_esmfold2fast (24s)" and only found by reading the JSON by hand afterwards.
  if [ ! -f "$OUT" ]; then
    echo "== $M FAILED: no result json at $OUT =="
    FAILED=$((FAILED + 1))
  elif ! python3 -c 'import json, sys
sys.exit(1 if json.load(open(sys.argv[1])).get("error") else 0)' "$OUT" 2>/dev/null; then
    echo "== $M FAILED: result json carries an error =="
    FAILED=$((FAILED + 1))
  fi
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,power.draw --format=csv,noheader \
    >> "$R/nvidia_${TAG}.txt"
done

echo "== session end $TAG: $(date -u +%FT%TZ) total=$(( $(date +%s) - START ))s =="
ls -l "$R"/*.json
[ "$FAILED" -eq 0 ] || { echo "== $FAILED row(s) failed =="; exit 1; }
