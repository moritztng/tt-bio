#!/bin/bash
# W1 -- name the roof for RELION's coarse pass, by attributing the getAllSquaredDifferencesCoarse
# region to symbols instead of assuming the region IS the kernel.
#
# This is a PROFILE, not a timed benchmark: it deliberately runs without benchlock and its output is
# a percentage split, not a wall. perf_event_paranoid is 4 on this box, which blocks unprivileged
# perf entirely; it is lowered for the profile and restored afterwards.
set -u
S=/home/ttuser/relion-scratch
T=$S/Tutorial5.0
BIN=$S/relion/build-e2e/bin/relion_refine_mpi
O=$S/w2/w1prof
mkdir -p "$O"

PARANOID_WAS=$(cat /proc/sys/kernel/perf_event_paranoid)
restore() { sudo -n sysctl -q -w kernel.perf_event_paranoid="$PARANOID_WAS" 2>/dev/null; }
trap restore EXIT
sudo -n sysctl -q -w kernel.perf_event_paranoid=1

cd "$T" || exit 1
# ref arm: no PYTHONPATH, so the TT bridge cannot load and RELION runs its own kernels.
mpirun -n 5 "$BIN" --o "$O/prof_run" \
  --continue "$T/Refine3D/job019/run_it012_optimiser.star" \
  --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images --iter 13 \
  > "$O/run.log" 2>&1 &
MPIPID=$!
echo "mpirun pid $MPIPID"

sleep "${WARM_S:-200}"                     # let it get past setup and into the particle loop

# one worker rank (rank 0 is the master and does no expectation work)
RANKS=$(pgrep -f "relion_refine_mpi --o $O/prof_run" | tr '\n' ' ')
echo "ranks: $RANKS"
TARGET=$(echo "$RANKS" | awk '{print $3}')
echo "profiling pid $TARGET for ${REC_S:-60}s"

perf record -F 499 -g --call-graph=fp -o "$O/perf.data" -p "$TARGET" -- sleep "${REC_S:-60}" \
  > "$O/record.log" 2>&1
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
  -p "$TARGET" -- sleep 20 > "$O/stat.txt" 2>&1

for p in $RANKS; do kill "$p" 2>/dev/null; done
kill "$MPIPID" 2>/dev/null
sleep 3
for p in $RANKS; do kill -9 "$p" 2>/dev/null; done

perf report -i "$O/perf.data" --no-children --percent-limit 0.4 --stdio 2>/dev/null \
  | head -60 > "$O/report.txt"
echo "=== perf report (self, >=0.4%) ==="; cat "$O/report.txt"
echo "=== perf stat ==="; cat "$O/stat.txt"
