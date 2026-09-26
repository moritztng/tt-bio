#!/usr/bin/env bash
# of3t-tapeamp: the dtype-reconciliation call census, on device, with the AICLK sampled DURING.

# One place decides what a valid AICLK is: perf/lib/aiclk.sh, mirroring tt_bio.aiclk.
_L=$(cd "$(dirname "$0")" && pwd); . "${_L%/perf/*}/perf/lib/aiclk.sh" || exit 1
set -uo pipefail
WT=/home/ttuser/.coworker/wt/of3t-tapeamp
cd "$WT"
SCR=/tmp/of3t/tapeamp
mkdir -p "$SCR"
PY=/home/ttuser/tt-bio-dev/env/bin/python
DEV=1
TAG=$1; shift
CLK=$SCR/aiclk_$TAG.log
: > "$CLK"
( while :; do
    echo "$(date +%s) $(aiclk "$DEV")" >> "$CLK"
    sleep 5
  done ) &
SIDE=$!
trap 'kill $SIDE 2>/dev/null' EXIT
L0=$(cut -d' ' -f1-3 /proc/loadavg)
S=$(date +%s)
echo "=== census $TAG start $(date -u +%FT%TZ) load[$L0] ==="
CENSUS_OUT="$WT/perf/of3t_tapeamp/CENSUS_$TAG.json" \
TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV \
  TT_BIO_LEASE_HOLDER=worker:of3t-tapeamp \
  TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$WT" \
  "$PY" perf/of3t_tapeamp/census_run.py perf/of3t_diffusion/device_gradient.py "$@"
RC=$?
E=$(date +%s)
L1=$(cut -d' ' -f1-3 /proc/loadavg)
kill $SIDE 2>/dev/null
CLKSTAT=$(awk '$2!="NA"{print $2}' "$CLK" | sort -n | awk '{a[n++]=$1; s+=$1} END{if(n==0){print "NA";exit} printf "n=%d min=%s med=%s max=%s mean=%.1f", n, a[0], a[int(n/2)], a[n-1], s/n}')
echo "=== census $TAG exit $RC seconds $((E-S)) load_start[$L0] load_end[$L1] aiclk_during{$CLKSTAT} $(date -u +%FT%TZ) ==="
exit $RC
