#!/usr/bin/env bash
# The token bucket's timing legs (perf_regression x3, then the 298 rung and the standing ladder),
# started only once the host is genuinely quiet. Two passes have now held these legs because qb2 was
# carrying another gate at loadavg 13-15, and each hold costs a whole launch. This waits instead.
#
# Three conditions, in order: the parity chain is gone (it owns the same card), the granted card is
# free, and loadavg has been under LOADMAX for three consecutive samples a minute apart. A single
# sample would fire in the gap between another gate's folds.
#
# perf_regression.py records no loadavg, which is why the same leg has read -16.0 % and -61.4 %
# hours apart with nothing in the artifact to tell them apart. So every leg's load is stamped here,
# before and after, into gate/timing_load.tsv.
#
#   CARD=3 setsid nohup bash perf/tokenbucket/timinglegs.sh > perf/tokenbucket/gate/timing.log 2>&1 &
set -u
: "${CARD:?set CARD to this launch grant}"
WT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$WT" || exit 1
LOADMAX=${LOADMAX:-5.0}
DEADLINE=$(( $(date +%s) + ${MAXWAIT_H:-8} * 3600 ))
TSV=perf/tokenbucket/gate/timing_load.tsv
mkdir -p perf/tokenbucket/gate
[ -f $TSV ] || printf 'when\tphase\tleg\tload1\tload5\n' > $TSV
stamp() { printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$(cut -d' ' -f1-2 /proc/loadavg | tr ' ' '\t')" >> $TSV; }

load1() { cut -d' ' -f1 /proc/loadavg; }
quiet_streak=0
while :; do
  now=$(date +%s)
  if [ "$now" -gt "$DEADLINE" ]; then
    echo "$(date -Is) GIVING UP: host never went quiet (load $(load1), limit $LOADMAX). Legs NOT run."
    exit 2
  fi
  if pgrep -f 'bash perf/tokenbucket/gate_tokenbucket.sh' >/dev/null; then
    echo "$(date -Is) waiting: the parity chain still owns card $CARD"; quiet_streak=0; sleep 60; continue
  fi
  holders=$(fuser /dev/tenstorrent/$CARD 2>/dev/null | tr -s ' ')
  if [ -n "$holders" ]; then
    echo "$(date -Is) waiting: card $CARD held by$holders"; quiet_streak=0; sleep 60; continue
  fi
  l=$(load1)
  if [ "$(echo "$l < $LOADMAX" | bc -l)" = 1 ]; then
    quiet_streak=$((quiet_streak + 1))
    echo "$(date -Is) quiet sample $quiet_streak/3 (load $l)"
    [ $quiet_streak -ge 3 ] && break
  else
    [ $quiet_streak -gt 0 ] && echo "$(date -Is) streak broken (load $l)"
    quiet_streak=0
  fi
  sleep 60
done

echo "$(date -Is) host quiet at load $(load1) — running the timing legs on card $CARD"
stamp begin all-timing-legs
CARD=$CARD LEGS=perf-protenix-v2,perf-opendde,perf-opendde-abag,sizeladder298,sizeladder \
  bash perf/tokenbucket/gate_tokenbucket.sh
rc=$?
stamp end all-timing-legs
echo "$(date -Is) timing legs finished rc=$rc"
