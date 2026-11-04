#!/usr/bin/env bash
# PNR TT cells on qb1 p150a UMD 0 (/dev/tenstorrent/1), per state/pnr/PROTOCOL.md (frozen 11-03 17:40Z).
#   run_cells.sh [--owner NAME] <out> [cells...]   cells: mol1 mol1_fast rfd4 bc2 (default all, in that order)
# --owner names the booking (default worker:pnr-tt); two chains booked at once need two owners, since book.py add
# replaces an owner's booking.
# Cells run in the order given (bc2 needs ~2 h 45 min, a Mol-1 cell ~10 min; Mol-1 first reaches the page sooner).
# The chain books all four cards once through ~/spd/feed/book.py as worker:pnr-tt for every cell it runs; each cell waits until the booking is
# active on every card, then takes all four ~/spd_qb1_card{0..3}.lock flocks (the locks gate legs and SPD chains use, so it
# queues behind a running gate and nothing new lands on qb1 during the cell), then waits for a device-free, quiet
# box (no other pid on any /dev/tenstorrent node, loadavg < 1.5) and runs under benchlock. Locks drop after each cell,
# the booking at the end. While it waits it drops fleet-pnr-tt, the card hold fleet.sh books for this row on every
# launch: that hold sits on one of the four cards, so the booking never reads active and lapses unused. tt_bio cells run ~/tt-bio-dev/env (ttnn 0.68.0, what main targets);
# /usr/bin/python3 imports 0.67.4. AICLK is sampled during every timed run by the harness itself.
set -u
WT=$HOME/.coworker/wt/pnr-tt; OWNER=worker:pnr-tt; [ "${1:-}" = --owner ] && { OWNER=$2; shift 2; }
OUT=${1:?out dir}; shift; CELLS=${*:-mol1 mol1_fast rfd4 bc2}
mkdir -p "$OUT"; exec >>"$OUT/chain.log" 2>&1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:pnr-tt
export BENCHLOCK_LOAD_WAIT_S=1800 BENCHLOCK_WAIT_S=21600
BL=$HOME/.coworker/scripts/benchlock.sh BOOK="python3 $HOME/spd/feed/book.py" PY=$HOME/tt-bio-dev/env/bin/python3
echo "start $(date -u +%FT%TZ) pid=$$ owner=$OWNER sha=$(git -C "$WT" rev-parse HEAD) cells=$CELLS"
holders() { for p in /proc/[0-9]*; do ls -l "$p/fd" 2>/dev/null | grep -q /dev/tenstorrent && echo "${p#/proc/}:$(cat "$p/comm" 2>/dev/null)"; done | tr "\n" " "; }
quiet() {
    local w=0 h l
    while h=$(holders); l=$(cut -d" " -f1 /proc/loadavg); [ -n "$h" ] || awk -v l="$l" "BEGIN{exit !(l >= 1.5)}"; do
        [ $((w % 900)) = 0 ] && echo "wait quiet (holders: ${h:-none}; load $l) $(date -u +%FT%TZ)"
        sleep 30; w=$((w+30)); [ $w -ge 43200 ] && { echo "GIVEUP quiet after 12h"; return 1; }
    done
    echo "quiet after ${w}s $(date -u +%FT%TZ) load $(cat /proc/loadavg)"
}
booked() { local c; for c in 0 1 2 3; do [ "$($BOOK active $c)" = "$OWNER" ] || return 1; done; }
secs() { case $1 in mol1|mol1_fast) echo 2700 ;; rfd4) echo 9000 ;; bc2) echo 11400 ;; esac; }
REM=0; for c in $CELLS; do REM=$((REM + $(secs $c))); done
cell() {  # name cmd...  (one booking covers the remaining cells, so no other booker lands between two of them)
    local n=$1; shift; local d="$OUT/$n" c; mkdir -p "$d"
    $BOOK rm fleet-pnr-tt > /dev/null; booked || echo "book $n+: $($BOOK add "$OWNER" 0,1,2,3 "$REM" --note "pnr-tt $CELLS (needs the box quiet)") $(date -u +%FT%TZ)"
    until booked; do sleep 60; $BOOK rm fleet-pnr-tt > /dev/null; $BOOK add "$OWNER" 0,1,2,3 "$REM" --note "pnr-tt $CELLS (needs the box quiet)" > /dev/null; done
    echo "booked $(date -u +%FT%TZ)"
    exec 5>>"$HOME/spd_qb1_card0.lock" 6>>"$HOME/spd_qb1_card1.lock" 7>>"$HOME/spd_qb1_card2.lock" 8>>"$HOME/spd_qb1_card3.lock"
    for c in 5 6 7 8; do flock -w 43200 $c || { echo "GIVEUP lock fd $c"; exec 5>&- 6>&- 7>&- 8>&-; return 1; }; done
    echo "locks held $(date -u +%FT%TZ)"
    quiet || { exec 5>&- 6>&- 7>&- 8>&-; $BOOK rm "$OWNER"; return 1; }
    { uptime; ~/.local/bin/tt-smi -s 2>/dev/null; } > "$d/host.start.txt"
    echo "cell $n start $(date -u +%FT%TZ)"
    ( cd "$WT" && "$BL" pnr-tt -- timeout 14400 "$@" ) > "$d/run.log" 2>&1 5>&- 6>&- 7>&- 8>&-
    echo "cell $n end rc=$? $(date -u +%FT%TZ) load $(cat /proc/loadavg)"
    exec 5>&- 6>&- 7>&- 8>&-; REM=$((REM - $(secs $n)))
}
for c in $CELLS; do case $c in
  mol1)      cell mol1 "$PY" perf/of3_4xpd/xmodel_ab.py --model mol1 --tree "$WT" --repeat 5 --label mol1 --out "$OUT/mol1/result.json" ;;
  mol1_fast) cell mol1_fast "$PY" perf/of3_4xpd/xmodel_ab.py --model mol1 --tree "$WT" --repeat 5 --fast --label mol1-fast --out "$OUT/mol1_fast/result.json" ;;
  rfd4)      cell rfd4 env PYTHONPATH="$WT" "$HOME/rfd4-fair/env/bin/python" perf/rfd4/fair/run.py \
                 --spec "$HOME/rfd4-inputs/pdl1-224.yaml" --weights "$HOME/rfd4-fair/bundle" --out "$OUT/rfd4" --part p150a \
                 --lane-width 10 --ladder-start 20 --ladder-step 10 --ceiling 200 --batches 5 ;;
  bc2)       cell bc2 env PYTHONPATH="$WT" JAPANFOLD_BC2_AF2_WEIGHTS="$HOME/bcx_e2e/af2_params" \
                 JAPANFOLD_BC2_MPNN_WEIGHTS="$HOME/bcx_e2e/bc2/bindcraft/weights/proteinmpnn/weights_neutral" \
                 "$HOME/bcx_e2e_venv/bin/python3" perf/spd/bench.py --out "$OUT/bc2" --chip 0 --model bindcraft2 \
                 --bc2 "$HOME/bcx_e2e/bc2" --seed 101 --warm 9 --inputs 80 ;;
esac; done
$BOOK rm "$OWNER"; echo "end $(date -u +%FT%TZ)"; touch "$OUT/done"
