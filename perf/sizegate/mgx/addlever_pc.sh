#!/usr/bin/env bash
# Splice TRIATT_SDPA_HIFI into the p150a size-ladder cells on pc chip 0, one fold per rung.
# The splice is refused for any model where another lever moved, so this is also a device check
# of the census half on the current tree. Timing is not re-measured.
set -u
cd "$(dirname "$0")/../../.."
root=$PWD log=$root/perf/sizegate/mgx/logs; mkdir -p "$log"
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=$root RELEASE_GATE_CENSUS_PYTHONPATH=$root
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:mgx-instrument
export RELEASE_GATE_SIZE_WORKDIR=$root/perf/sizegate/work-addlever RELEASE_GATE_FOLD_TIMEOUT=3600
py=$HOME/tt-bio/env/bin/python
for m in ${MODELS:-boltz2 protenix-v1 nesso1 esmfold2 openbind openfold3 protenix-v2 opendde rf3}; do
  echo "[$(date -u +%FT%TZ)] START addlever $m" >> "$log/addlever-p150a.log"
  "$py" scripts/release_gate.py --model size-ladder --size-ladder-models "$m" \
        --size-ladder-record-lever TRIATT_SDPA_HIFI >> "$log/addlever-p150a.log" 2>&1
  rc=$?
  echo "[$(date -u +%FT%TZ)] EXIT $rc addlever $m" >> "$log/addlever-p150a.log"
done
