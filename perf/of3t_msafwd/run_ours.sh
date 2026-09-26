#!/usr/bin/env bash
# of3t-msafwd: one device arm on qb1 card 1 (p150a). host_quiet.py green before (logged), AICLK
# sampled DURING every 2 s, one line per arm in ARMS_device.tsv. Accuracy only: no timing is read.

# One place decides what a valid AICLK is: perf/lib/aiclk.sh, mirroring tt_bio.aiclk.
_L=$(cd "$(dirname "$0")" && pwd); . "${_L%/perf/*}/perf/lib/aiclk.sh" || exit 1
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-msafwd
cd "$W" || exit 1
S=/home/ttuser/of3t_msafwd
PY=/home/ttuser/tt-bio-dev/env/bin/python
DEV=1
export PYTHONPATH="/home/ttuser/of3t-campaign-refs/of3pkg043:/home/ttuser/of3t_frame384/ref:$W/perf/of3t_tape:$W"
export OMP_NUM_THREADS=8
export TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV TT_BIO_LEASE_HOLDER=worker:of3t-msafwd
TAG=$1; shift
HQ=perf/c12_orchestrator/pair_guard/host_quiet.py
for i in $(seq 1 60); do python3 $HQ --quiet && break; sleep 10; done
python3 $HQ --quiet; Q0=$?
[ $Q0 -eq 0 ] || { echo "host_quiet RED before $TAG"; exit 9; }
BOARD=$(cat /sys/class/tenstorrent/tenstorrent\!$DEV/tt_card_type)
echo "host_quiet GREEN before ours_$TAG load[$(cut -d' ' -f1-3 /proc/loadavg)] host=$(hostname) board=$BOARD card=$DEV $(date -u +%FT%TZ)" | tee -a perf/of3t_msafwd/HOST_QUIET.txt
CLK=$S/aiclk_$TAG.log; : > "$CLK"
( while :; do echo "$(date +%s) $(aiclk "$DEV")" >> "$CLK"; sleep 2; done ) &
SIDE=$!
trap 'kill $SIDE 2>/dev/null' EXIT
"$PY" "$@"
RC=$?; kill $SIDE 2>/dev/null
CLKSTAT=$(awk '$2!="NA"{print $2}' "$CLK" | sort -n | awk '{a[n++]=$1; s+=$1} END{if(n==0){print "NA";exit} printf "n=%d min=%s med=%s max=%s mean=%.1f", n, a[0], a[int(n/2)], a[n-1], s/n}')
echo "=== ours $TAG exit $RC host=$(hostname) board=$BOARD card=$DEV aiclk_during{$CLKSTAT} $(date -u +%FT%TZ) ==="
printf '%s\n' "$TAG $RC host=$(hostname) board=$BOARD card=$DEV q$Q0 \"$CLKSTAT\"" >> perf/of3t_msafwd/ARMS_device.tsv
exit $RC
