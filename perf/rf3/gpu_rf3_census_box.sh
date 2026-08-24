#!/bin/bash
# The whole rental, unattended. Every step writes a .ok marker and is skipped if it exists, so a
# relaunched agent resumes instead of paying twice.
#
#   bash /work/repo/perf/rf3/gpu_rf3_census_box.sh all
#
# Arms, in the order they run. The order matters: S5 changes the installed cuEquivariance ops
# wheel, so it is last and nothing measured before it can be contaminated by it.
#
#   S1  setup      python 3.12 venv, `rc-foundry[rf3]`, checkpoint, CCD warm
#   S2  census     PRIMARY ARM. Shipped defaults, 512 aa, 10 recycles, 50 steps, every module
#                  forward-hooked. This is the deliverable.
#   S3  profile    Labelled contrast. Same defaults but 1 recycle / 2 steps, under torch.profiler,
#                  so the CUDA kernel NAMES answer "generic or sm100f" instead of a wheel A/B.
#   S4  no-cueq    Labelled contrast. Forces the vanilla PyTorch triangle path, to read the dtypes
#                  of the implementation RF3 does not ship.
#   S5  cu13       Labelled contrast, Blackwell only. Installs cuequivariance-ops-torch-cu13 and
#                  re-profiles, because the sm100f triangle-attention kernel ships only in the cu13
#                  wheels (cuEquivariance changelog 0.8.0 and 0.10.0).
set -u
R=/work/results
mkdir -p $R /work/out
exec >>$R/master.log 2>&1
PY=/work/v_rf3/bin/python
CEN=/work/repo/perf/rf3/gpu_rf3_dtype_census.py
IN=/work/repo/perf/rf3/inputs/rf3_512.json
say() { echo "=== $(date -u +%FT%TZ) $*"; }
smi() { nvidia-smi --query-compute-apps=pid,used_gpu_memory,process_name --format=csv; \
        nvidia-smi --query-gpu=name,compute_cap,power.draw,power.limit,memory.used,driver_version \
                   --format=csv; }

step_setup() {
  [ -f $R/S1.ok ] && { say "S1 skip"; return; }
  say "S1 setup"
  bash /work/repo/perf/rf3/gpu_rf3_setup.sh
  [ -f /work/SETUP_OK ] || { say "S1 FAILED"; return 1; }
  $PY -m pip list 2>/dev/null | tee $R/pip_list.txt | grep -iE 'cuequivariance|torch|foundry|atomworks|triton|lightning'
  sha256sum $IN | tee $R/input_sha256.txt
  smi > $R/smi_before.txt
  touch $R/S1.ok
}

step_census() {
  [ -f $R/S2.ok ] && { say "S2 skip"; return; }
  say "S2 census PRIMARY (shipped defaults, 512 aa, 10 recycles, 50 steps, seed 42)"
  $PY $CEN --inputs $IN --out-dir /work/out/census --report $R/census.json \
      --n-recycles 10 --num-steps 50 --diffusion-batch-size 1 --seed 42 \
      --early-stop-plddt 0 --label census-512-defaults || return 1
  touch $R/S2.ok
}

step_profile() {
  [ -f $R/S3.ok ] && { say "S3 skip"; return; }
  say "S3 profiler arm (1 recycle, 2 steps) -- kernel names"
  $PY $CEN --inputs $IN --out-dir /work/out/prof --report $R/prof_default.json \
      --n-recycles 1 --num-steps 2 --diffusion-batch-size 1 --seed 42 \
      --early-stop-plddt 0 --profile --label profile-default-wheel || return 1
  touch $R/S3.ok
}

step_nocueq() {
  [ -f $R/S4.ok ] && { say "S4 skip"; return; }
  say "S4 contrast arm: vanilla triangle path"
  $PY $CEN --inputs $IN --out-dir /work/out/nocueq --report $R/census_nocueq.json \
      --n-recycles 1 --num-steps 2 --diffusion-batch-size 1 --seed 42 \
      --early-stop-plddt 0 --no-cueq --label contrast-no-cueq || return 1
  touch $R/S4.ok
}

step_cu13() {
  [ -f $R/S5.ok ] && { say "S5 skip"; return; }
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
  if [ "${CC:-0}" -lt 100 ]; then
    say "S5 skip: compute cap $CC is not Blackwell, the sm100f kernel cannot exist here"
    echo "skipped: compute_cap=$CC" > $R/S5.skipped
    touch $R/S5.ok
    return
  fi
  say "S5 cu13 ops wheel arm (Blackwell only)"
  $PY -m pip install -q "cuequivariance-ops-torch-cu13==$($PY -c "
from importlib.metadata import version
print(version('cuequivariance-torch'))")" || { say "S5 wheel install failed"; return 1; }
  $PY -m pip list 2>/dev/null | grep -i cuequivariance > $R/pip_list_cu13.txt
  $PY $CEN --inputs $IN --out-dir /work/out/prof13 --report $R/prof_cu13.json \
      --n-recycles 1 --num-steps 2 --diffusion-batch-size 1 --seed 42 \
      --early-stop-plddt 0 --profile --label profile-cu13-wheel || return 1
  touch $R/S5.ok
}

case "${1:-all}" in
  all) step_setup && step_census && step_profile && step_nocueq && step_cu13 \
       && { smi > $R/smi_after.txt; say "ALL DONE"; touch $R/ALL_DONE; } ;;
  *)   "step_$1" ;;
esac
