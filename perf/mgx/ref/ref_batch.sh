#!/usr/bin/env bash
# Fold MGX reference cells on the rented box, one process per cell, skipping any cell whose
# record already says ok. Each model runs in its own upstream venv (see ref_fold.py's map).
#
#   bash perf/mgx/ref/ref_batch.sh "<model>" "<fixture> ..." "<seed> ..."
#   bash perf/mgx/ref/ref_batch.sh all all "0 1"
#
# Run from the repo root. Records land in ${MGX_OUT:-/root/refs}/<model>/<fixture>/s<seed>.json.
# MGX_ONLY=oom re-folds only cells whose record says oom (with MGX_DTYPE=bf16 for the protenix
# family's fallback); by default every cell that is not ok is (re)folded.
set -u
cd "$(dirname "$0")/../../.."
OUT=${MGX_OUT:-/root/refs}
models=$1; fixtures=$2; seeds=${3:-"0 1"}
[ "$models" = all ] && models=$(python3 -c "import json;print(' '.join(json.load(open('perf/mgx/ref/plan.json'))['models']))")
[ "$fixtures" = all ] && fixtures=$(python3 -c "import json;print(' '.join(json.load(open('perf/mgx/ref/fixtures/fixtures.json'))['fixtures']))")

venv() {
  case "$1" in
    boltz2) echo /root/venv-boltz ;;
    protenix-v1|protenix-v2) echo /root/venv-protenix ;;
    opendde|opendde-abag) echo /root/venv-opendde ;;
    openfold3) echo /root/venv-of3 ;;
    openbind) echo /root/venv-ob ;;
    esmfold2|esmfold2-fast) echo /root/venv-esm312 ;;
    rf3) echo /root/venv-rf3 ;;
    *) echo "no venv for $1" >&2; return 1 ;;
  esac
}

for f in $fixtures; do
  for m in $models; do
    for s in $seeds; do
      rec="$OUT/$m/$f/s$s.json"
      if [ -s "$rec" ] && grep -q '"status": "ok"' "$rec"; then continue; fi
      if [ -n "${MGX_ONLY:-}" ] && ! grep -q "\"status\": \"$MGX_ONLY\"" "$rec" 2>/dev/null; then continue; fi
      v=$(venv "$m") || continue
      sp=$("$v/bin/python" -c "import site;print(site.getsitepackages()[0])")
      # torch cu13 wheels JIT-compile a few kernels through nvrtc and need its builtins on the
      # loader path; the cuEquivariance ops library is not found without it either.
      export LD_LIBRARY_PATH="$sp/nvidia/cu13/lib:$sp/cuequivariance_ops/lib:${BASE_LD:-}"
      mkdir -p "$OUT/$m/$f"
      echo "== $m $f s$s $(date -u +%FT%TZ)"
      "$v/bin/python" perf/mgx/ref/ref_fold.py --model "$m" --fixture "$f" --seed "$s" \
        --out "$OUT" --dtype "${MGX_DTYPE:-fp32}" > "$OUT/$m/$f/s$s.log" 2>&1
      tail -1 "$OUT/$m/$f/s$s.log"
    done
  done
done
echo "BATCH_DONE $(date -u +%FT%TZ)"
