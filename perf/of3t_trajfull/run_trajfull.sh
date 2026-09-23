#!/bin/bash
# of3t-trajfull: the device census, then the one 20-rung arm, on card 0.
#
# `run_refatom.sh` with two differences:
#   * `--namemap structural`, so the dump carries all 761 reference names rather than 581;
#   * a `census` mode that stops after the discovery walk, because a wrong map has to fail
#     on a two-minute run and not after half an hour of device time.
# The reference side is NOT re-run. `w/theirs/k20.npz` is complete at 761 tensors and is the
# only thing in this campaign that is.
set -u
MODE=${1:-arm}
CARD=${2:-0}
PY=/home/ttuser/tt-bio-dev/env/bin/python
R=/home/ttuser/of3t_runs/trajwide
ARM=trajfull
mkdir -p "$R"
C="$R/chain_${ARM}.log"
say() { echo "$(date -u +%FT%TZ) [card $CARD] $*" >> "$C"; }

export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-trajfull
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

clock_watch() {
  while :; do
    printf "%s " "$(date -u +%FT%TZ)"
    ~/.local/bin/tt-smi -s 2>/dev/null | tr -d " " | grep -i aiclk | head -4 | tr "\n" " "
    printf "\n"; sleep 60
  done
}

exec 9>"$R/$ARM.lock"
if ! flock -n 9; then say "already running, exiting"; exit 0; fi
say "start mode=$MODE pid=$$ cwd=$PWD"

if [ "$MODE" = census ]; then
  $PY perf/of3t_trajwide/trajwide.py --side ours --arm "$ARM" --refatom device --threads 3 \
      --namemap structural --census-out perf/of3t_trajfull/DEVICE_CENSUS.json \
      > "$R/census_$ARM.log" 2> "$R/census_$ARM.err"
  RC=$?
  say "census done rc=$RC"
  echo "mode=census rc=$RC card=$CARD at=$(date -u +%FT%TZ)" > "$R/$ARM.census.done"
  exit $RC
fi

echo "=== $ARM start $(date -u +%FT%TZ) card $CARD" >> "$R/ours_$ARM.log"
clock_watch >> "$R/aiclk_$ARM.log" 2>&1 &
CW=$!
$PY perf/of3t_trajwide/trajwide.py --side ours --arm "$ARM" --refatom device --threads 3 \
    --namemap structural \
    >> "$R/ours_$ARM.log" 2>> "$R/ours_$ARM.err"
RC=$?
kill "$CW" 2>/dev/null
echo "=== $ARM done rc=$RC $(date -u +%FT%TZ)" >> "$R/ours_$ARM.log"
echo "arm=$ARM rc=$RC card=$CARD at=$(date -u +%FT%TZ)" > "$R/$ARM.done"
say "done rc=$RC"
exec 9>&-
exit $RC
