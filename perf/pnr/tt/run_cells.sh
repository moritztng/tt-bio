#!/usr/bin/env bash
# PNR TT cells on qb1 p150a UMD 0 (/dev/tenstorrent/1), per state/pnr/PROTOCOL.md (frozen 11-03 17:40Z).
#   run_cells.sh <out> [cells...]   cells: mol1 mol1_fast rfd4 bc2 (default all, in that order)
# Each cell first waits for a device-free, quiet box (no other pid on any /dev/tenstorrent node, loadavg < 1.5),
# then runs under benchlock. AICLK is sampled during every timed run by the harness itself.
set -u
WT=$HOME/.coworker/wt/pnr-tt; OUT=${1:?out dir}; shift; CELLS=${*:-mol1 mol1_fast rfd4 bc2}
mkdir -p "$OUT"; exec >>"$OUT/chain.log" 2>&1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:pnr-tt
export BENCHLOCK_LOAD_WAIT_S=1800 BENCHLOCK_WAIT_S=21600
BL=$HOME/.coworker/scripts/benchlock.sh
echo "start $(date -u +%FT%TZ) pid=$$ sha=$(git -C "$WT" rev-parse HEAD) cells=$CELLS"
holders() { for p in /proc/[0-9]*; do ls -l "$p/fd" 2>/dev/null | grep -q /dev/tenstorrent && echo "${p#/proc/}:$(cat "$p/comm" 2>/dev/null)"; done | tr "\n" " "; }
quiet() {
    local w=0 h l
    while h=$(holders); l=$(cut -d" " -f1 /proc/loadavg); [ -n "$h" ] || awk -v l="$l" "BEGIN{exit !(l >= 1.5)}"; do
        [ $((w % 900)) = 0 ] && echo "wait quiet (holders: ${h:-none}; load $l) $(date -u +%FT%TZ)"
        sleep 30; w=$((w+30)); [ $w -ge 43200 ] && { echo "GIVEUP quiet after 12h"; return 1; }
    done
    echo "quiet after ${w}s $(date -u +%FT%TZ) load $(cat /proc/loadavg)"
}
cell() {  # name cmd...
    local n=$1; shift; local d="$OUT/$n"; mkdir -p "$d"
    quiet || return 1
    { uptime; ~/.local/bin/tt-smi -s 2>/dev/null; } > "$d/host.start.txt"
    echo "cell $n start $(date -u +%FT%TZ)"
    ( cd "$WT" && "$BL" pnr-tt -- timeout 14400 "$@" ) > "$d/run.log" 2>&1
    echo "cell $n end rc=$? $(date -u +%FT%TZ) load $(cat /proc/loadavg)"
}
for c in $CELLS; do case $c in
  mol1)      cell mol1 python3 perf/of3_4xpd/xmodel_ab.py --model mol1 --tree "$WT" --repeat 5 --label mol1 --out "$OUT/mol1/result.json" ;;
  mol1_fast) cell mol1_fast python3 perf/of3_4xpd/xmodel_ab.py --model mol1 --tree "$WT" --repeat 5 --fast --label mol1-fast --out "$OUT/mol1_fast/result.json" ;;
  rfd4)      cell rfd4 env PYTHONPATH="$WT" "$HOME/rfd4-fair/env/bin/python" perf/rfd4/fair/run.py \
                 --spec "$HOME/rfd4-inputs/pdl1-224.yaml" --weights "$HOME/rfd4-fair/bundle" --out "$OUT/rfd4" --part p150a \
                 --lane-width 10 --ladder-start 20 --ladder-step 10 --ceiling 200 --batches 5 ;;
  bc2)       cell bc2 env PYTHONPATH="$WT" JAPANFOLD_BC2_AF2_WEIGHTS="$HOME/bcx_e2e/af2_params" \
                 JAPANFOLD_BC2_MPNN_WEIGHTS="$HOME/bcx_e2e/bc2/bindcraft/weights/proteinmpnn/weights_neutral" \
                 "$HOME/bcx_e2e_venv/bin/python3" perf/spd/bench.py --out "$OUT/bc2" --chip 0 --model bindcraft2 \
                 --bc2 "$HOME/bcx_e2e/bc2" --seed 101 --warm 9 --inputs 80 ;;
esac; done
echo "end $(date -u +%FT%TZ)"; touch "$OUT/done"
