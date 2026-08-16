#!/usr/bin/env bash
# Is the boltz2-9ncy-nomsa drift ours? Same leg, three ways, fresh workdir each time so nothing
# resumes: dual-NOC off (D disabled globally), dual-NOC on (the shipped default), and a repeat of
# the shipped default to see whether the leg is even stable run to run.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p2
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p2
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3

for arm in dnoff dnon dnon_repeat; do
  case "$arm" in
    dnoff) export TT_BIO_TRIMUL_DUAL_NOC=0 ;;
    *)     unset TT_BIO_TRIMUL_DUAL_NOC ;;
  esac
  echo "########## arm=$arm TT_BIO_TRIMUL_DUAL_NOC=${TT_BIO_TRIMUL_DUAL_NOC:-<default>} ##########"
  rm -rf "$WT/.iso_$arm"
  $PY -u scripts/full_parity_gate.py --leg boltz2-9ncy-nomsa \
    --workers localhost:0 --workdir "$WT/.iso_$arm" 2>&1 | grep -E "^boltz2-9ncy|Tally"
  cp "$WT/.iso_$arm/boltz2-9ncy-nomsa.json" "$WT/perf/esmbeat/iso_9ncy_$arm.json" 2>/dev/null
done
echo "ISO_DONE"
