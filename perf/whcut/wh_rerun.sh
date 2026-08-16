#!/bin/bash
# Re-run what the card race broke (§100), waiting on the holder's PID rather than on card
# occupancy. Occupancy is the wrong signal: the parity gate releases and re-acquires card 28
# between seeds, so a 60-second poller reads "free" mid-run and starts on top of it. That is
# exactly how clash_sweep and the gate destroyed each other's work.
set -u
TREE=/home/cust-team/mthuening/whbase/wt-whcut
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python3
OUT=$TREE/perf/whcut/out
SWEEP_PID=${SWEEP_PID:-1259755}
cd "$TREE" || exit 1
while kill -0 "$SWEEP_PID" 2>/dev/null; do sleep 30; done
sleep 20   # let the last fold's device handle close before opening it again
echo "WH RERUN START $(date -u +%FT%TZ)"

# protenix-hsa-msa: a served model at 585 aa, which pads to 640 and so sits in K3's band.
# It is the protenix counterpart of boltz2-hsa-nomsa and the only served-model leg the gate
# did not score. boltzgen is NOT re-run: it is an in-process leg that ignores --workers and
# opens production card 0, so it cannot run on this box at all (§100).
ESM_ROOT=/home/cust-team/mthuening/esm \
RELEASE_GATE_MSA_DIR=/home/cust-team/mthuening/abag_xm/msa_cache \
PYTHONPATH="$TREE" TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/full_parity_gate.py --workers UF-EV-A13-GWH02:28 --fresh \
    --fold-timeout 4800 --workdir "$OUT/parity-wh-rerun" --leg protenix-hsa-msa
echo "PROTENIX HSA RERUN EXIT $? $(date -u +%FT%TZ)"
