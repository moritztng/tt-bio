#!/usr/bin/env bash
# esmc-300m read -22.5% on a WARM-MEDIAN leg, the class perf_regression.py measures at under 0.5%
# run-to-run, so draw noise does not explain it and it needs a control rather than a shrug.
# The leg is cheap (2 s load, 5 timed calls at ~67 ms), so four alternating arms cost ~2 minutes.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge
cd "$WT" || exit 1
OUT="$WT/perf/ttx_a3/gate2/esmc_attr"
PROG="$OUT/progress"
P=/home/ttuser/tt-bio-dev/env/bin/python3
BENCH=/home/ttuser/.coworker/scripts/benchlock.sh
mkdir -p "$OUT"; touch "$PROG"
export PYTHONPATH="$WT" TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }
rep() {
  local name="$1" arm="$2"
  grep -q " $name rc=" "$PROG" && { echo "skip $name"; return 0; }
  local ev=() l0 l1
  [ "$arm" = off ] && ev=(TT_BIO_SDPA_FUSED_LARGE_S=0)
  l0=$(cut -d' ' -f1 /proc/loadavg)
  env "${ev[@]}" TT_VISIBLE_DEVICES=0 timeout 900 bash "$BENCH" \
    "worker:ttx-a3-sdpa-ship-remerge" -- "$P" scripts/perf_regression.py --model esmc-300m \
    > "$OUT/$name.log" 2>&1
  local rc=$?
  l1=$(cut -d' ' -f1 /proc/loadavg)
  log "$name rc=$rc arm=$arm load0=$l0 load1=$l1 | $(grep -oE 'esmc-300m +seq/s +[0-9.]+ +[0-9.]+ +[-+][0-9.]+%' "$OUT/$name.log" | tail -1)"
}
rep e_off1 off
rep e_on1  on
rep e_off2 off
rep e_on2  on
log "ESMC_ATTR_DONE"
cat "$PROG"
