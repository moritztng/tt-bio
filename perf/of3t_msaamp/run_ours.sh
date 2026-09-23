#!/usr/bin/env bash
# of3t-msaamp: our device arm at the msa_module boundary, qb1 card 1 (p150a), with the gradient
# tensors dumped so they are scored by the same function as the upstream arms. AICLK sampled
# DURING every 2 s; host_quiet.py before and after.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-msaamp
cd "$W" || exit 1
S=/home/ttuser/of3t_msaamp
PY=/home/ttuser/tt-bio-dev/env/bin/python
DEV=1
export PYTHONPATH="$W/perf/of3t_tape:$W"
export OMP_NUM_THREADS=8
export TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV TT_BIO_LEASE_HOLDER=worker:of3t-msaamp
TAG=$1; shift
CLK=$S/aiclk_$TAG.log; : > "$CLK"
python3 perf/c12_orchestrator/pair_guard/host_quiet.py; Q0=$?
( while :; do echo "$(date +%s) $(cat /sys/class/tenstorrent/tenstorrent\!$DEV/tt_aiclk 2>/dev/null || echo NA)" >> "$CLK"; sleep 2; done ) &
SIDE=$!
trap 'kill $SIDE 2>/dev/null' EXIT
echo "=== ours $TAG board=$(cat /sys/class/tenstorrent/tenstorrent\!$DEV/tt_card_type) card=$DEV host=$(hostname) load[$(cut -d' ' -f1-3 /proc/loadavg)] host_quiet_before=$Q0 $(date -u +%FT%TZ) ==="
T0=$(date +%s)
"$PY" perf/of3t_auxheads/msa_instrument.py \
  --boundary $S/cap043b/boundary_msa_module.pt \
  --reference-grads /home/ttuser/of3t-campaign-refs/bundle_min_043/grads_f64_043.pt \
  --dump-grads "$S/grads_$TAG.pt" "$@" --out "perf/of3t_msaamp/$TAG.json"
RC=$?; T1=$(date +%s); kill $SIDE 2>/dev/null
python3 perf/c12_orchestrator/pair_guard/host_quiet.py --quiet; Q1=$?
CLKSTAT=$(awk '$2!="NA"{print $2}' "$CLK" | sort -n | awk '{a[n++]=$1; s+=$1} END{if(n==0){print "NA";exit} printf "n=%d min=%s med=%s max=%s mean=%.1f", n, a[0], a[int(n/2)], a[n-1], s/n}')
echo "=== ours $TAG exit $RC seconds $((T1-T0)) load_end[$(cut -d' ' -f1-3 /proc/loadavg)] host_quiet_after=$Q1 aiclk_during{$CLKSTAT} $(date -u +%FT%TZ) ==="
printf '%s\n' "$TAG $RC $((T1-T0)) q$Q0/q$Q1 \"$CLKSTAT\"" >> "$W/perf/of3t_msaamp/ARMS_device.tsv"
exit $RC
