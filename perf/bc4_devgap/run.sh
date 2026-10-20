#!/bin/bash
# bc4-bet-devgap: one device-profiled BindCraft 2 run on a qb2 chip, detached.
# usage: CHIP=3 perf/bc4_devgap/run.sh <out-name> [devprof args...]   marker <out>/done
set -u
NAME=$1; shift
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT" || exit 1
OUT=$WT/_devgap/$NAME; mkdir -p "$OUT"
C=${CHIP:?}; T=${TRACY_METAL:-$HOME/tt-metal-b2z}
git rev-parse HEAD > "$OUT/sha"; git diff > "$OUT/diff.patch"; cat /proc/loadavg > "$OUT/loadavg.start"
export TT_METAL_HOME=$T PYTHONPATH=$T/ttnn:$T/tools:$T:$WT
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bc4-bet-devgap
export JAPANFOLD_BC2_AF2_WEIGHTS=$HOME/bcx_e2e/af2_params
export TT_BIO_TRACE_RECOMPUTE=${TRACE:-1} TT_BIO_TRACE_REGION_MB=2048
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
export TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 TT_METAL_TRACE_PROFILER=1
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=${SUPPORT:-12000}
unset LD_LIBRARY_PATH
flock -w 60 $HOME/spd_qb2_chip$C.lock timeout ${TMO:-5400} $HOME/bcx_e2e_venv/bin/python3 \
    perf/bc4_devgap/devprof.py --out "$OUT" "$@" > "$OUT/devprof.log" 2>&1
echo "rc=$? $(date -u +%FT%TZ) $(cat /proc/loadavg)" > "$OUT/done"
