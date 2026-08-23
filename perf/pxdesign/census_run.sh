#!/usr/bin/env bash
# Census the PXDesign CLI's token axis, and prove the CLI end to end in the same run.
#
# Detached and long-bounded on purpose: the counters come from a real job (tt_bio/token_axis.py's
# own rule) and on a loaded box the Protenix trunk plus first-call kernel compilation is tens of
# minutes before the first wrapped call. Output is unbuffered to a log rather than piped, so
# progress is readable while it runs instead of only at exit.
#
#   CENSUS_CARD=3 bash perf/pxdesign/census_run.sh
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CARD="${CENSUS_CARD:?set CENSUS_CARD to a card this task holds}"
PY=/home/ttuser/tt-bio-dev/env/bin/python3
DIR="${CENSUS_DIR:-/tmp/pxd_census}"
OUT="${CENSUS_OUT:-/tmp/pxd_cli_e2e}"
cd "$WT" || exit 1
rm -rf "$DIR" "$OUT"; mkdir -p "$DIR" logs

export TOKEN_AXIS_CENSUS_DIR="$DIR"
export PYTHONPATH="$WT/perf/bucketing_audit/censusenv:$WT"
export PXDESIGN_CKPT="${PXDESIGN_CKPT:-/home/ttuser/pxd_ckpt/pxdesign_v0.1.0.pt}"
export TT_VISIBLE_DEVICES="$CARD"
export TT_BIO_LEASE_CARDS="1,$CARD"
export TT_BIO_LEASE_HOLDER=worker:flight-land-pxdesign-af2ig

# n_step is low because the step count changes how MANY times a site is reached, not WHICH sites,
# and it is also what the release-gate design leg wants.
echo "=== $(date -Is) census start on card $CARD; loadavg $(cut -d' ' -f1-3 /proc/loadavg) ==="
timeout 3600 "$PY" -u -m tt_bio.main design tests/fixtures/pxdesign/PDL1.yaml \
    --model pxdesign --out_dir "$OUT" --n_step 20 --seed 0
echo "rc=$?"
echo "=== $(date -Is) census done; loadavg $(cut -d' ' -f1-3 /proc/loadavg) ==="
ls -la "$OUT" 2>/dev/null
PYTHONPATH="$WT" "$PY" tests/token_axis_probe.py "$DIR"
