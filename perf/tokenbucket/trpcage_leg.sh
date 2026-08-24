#!/usr/bin/env bash
# The accuracy leg that came back from the dead. opendde-trpcage-nomsa had been
# BLOCKED-REF-REGEN-NEEDED since it was harvested, because the harvest script named the reference CIF
# after the target and the gate looks for the input yaml's name. Main fixed that today (28c9f16c) and
# put the CIFs on the parity-fixtures-latest release asset.
#
# It matters more than its wall clock suggests: its input is trpcage at 20 aa, the most extreme
# relative pad in the whole gate (20 -> 64), which is the exact rung the perf argument for this flip
# turns on. Nothing else this branch has run scores accuracy there.
set -u
: "${CARD:?set CARD to this launch grant}"
WT=/home/ttuser/.coworker/wt/tokenbucket-rebase-and-land
SLUG=tokenbucket-rebase-and-land
PY=${GATE_PYTHON:-/home/ttuser/.coworker/rel070/relvenv/bin/python3}
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
export PATH=/home/ttuser/.local/bin:/home/ttuser/tt-bio/env/bin:$PATH
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:$SLUG"
OUT=perf/tokenbucket/trpcage
mkdir -p "$OUT"

echo "=== $(date -Is) fetch the parity fixture asset (idempotent, only overwrites what the tarball has)"
bash scripts/fetch_parity_fixtures.sh > "$OUT/fetch.log" 2>&1
echo "fetch rc=$? $(tail -2 "$OUT/fetch.log")"
ls -la docs/implementation-parity-data/ref-fixtures/opendde/trpcage/nomsa_4cycle_20step_1sample_fp32_reduced/seed0/ 2>&1 | tail -6

echo "=== $(date -Is) opendde-trpcage-nomsa on card $CARD"
env $LEASE PYTHONPATH=$WT ESM_ROOT=$ESM_ROOT \
  "$PY" -u scripts/full_parity_gate.py --workdir "/tmp/tb_trpcage_$(git rev-parse --short HEAD)" \
    --workers "qb2:$CARD" --leg opendde-trpcage-nomsa > "$OUT/leg.log" 2>&1
echo "=== $(date -Is) leg rc=$? load $(cut -d' ' -f1-3 /proc/loadavg)"
grep -E "PASS|FAIL|BLOCKED|GATE|verdict|leg " "$OUT/leg.log" | tail -15
