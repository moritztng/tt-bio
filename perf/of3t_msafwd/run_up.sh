#!/usr/bin/env bash
# of3t-msafwd: one upstream 0.4.3 site-by-site arm. CPU only, no card, no lease, behind a green
# host_quiet.py (logged to HOST_QUIET.txt).
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-msafwd
cd "$W" || exit 1
export PYTHONPATH="/home/ttuser/of3t-campaign-refs/of3pkg043:/home/ttuser/of3t_frame384/ref:/home/ttuser/of3t_hostleg/deps:/home/ttuser/of3t_hostleg/pylibs:$W"
export OMP_NUM_THREADS=8
PY=/home/ttuser/tt-bio-dev/env/bin/python
S=/home/ttuser/of3t_msafwd
BND=${BND:-/home/ttuser/of3t_msaamp/crop64.pt}
TAG=$1; shift
HQ=perf/c12_orchestrator/pair_guard/host_quiet.py
for i in $(seq 1 60); do python3 $HQ --quiet && break; sleep 10; done
python3 $HQ --quiet || { echo "host_quiet RED before $TAG"; exit 9; }
echo "host_quiet GREEN before up_$TAG load[$(cut -d' ' -f1-3 /proc/loadavg)] host=$(hostname) $(date -u +%FT%TZ)" | tee -a perf/of3t_msafwd/HOST_QUIET.txt
nice -n 5 "$PY" perf/of3t_msafwd/up_fwd.py --boundary "$BND" "$@" --report "perf/of3t_msafwd/UP_$TAG.json"
echo "=== up $TAG exit $? $(date -u +%FT%TZ) ==="
