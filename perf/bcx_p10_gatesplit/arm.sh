#!/bin/bash
# One arm of the gate-split round: two BindCraft 2 trajectories in one process, full stack5
# configuration, the gate held across the whole device region (split 0) or for the enqueue only
# (split 1, `duotraj.DeviceGate.split`).
#   arm.sh <tag> <rounds> <serial|duo> <split 0|1>
# Runs on pc (card 0) or qb2 (TT_VISIBLE_DEVICES from the caller); the host picks the paths.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; mode=$3; split=$4; shift 4
case "$mode" in
    serial) duo=0 ;;
    duo)    duo=1 ;;
    *) echo "arm.sh: mode must be serial or duo, got '$mode'" >&2; exit 2 ;;
esac
if [ "$rounds" -gt 9 ]; then echo "arm.sh: 9 rounds is the ceiling, see bcx-p10-rne" >&2; exit 2; fi
out=perf/bcx_p10_gatesplit/out/$tag
rm -rf "$out"; mkdir -p "$out"
case $(hostname) in
    tt-quietbox2*) root=/home/ttuser/bcx_e2e; py=/home/ttuser/bcx_e2e_venv/bin/python3
                   params=$root/af2_params; card=${TT_VISIBLE_DEVICES:?pin a qb2 card} ;;
    *)             root=/home/moritz/bcx_shipped; py=/home/moritz/bcx_hostcut_venv/bin/python3
                   params=$root/af2_params; card=0 ;;
esac
export PYTHONPATH=$PWD
export BCX_BC2=$root/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_gatesplit/out/xlacache
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-$card}
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-gatesplit
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
export TT_BIO_GRAD_FANIN_L1=0 TT_BIO_GENQ_COMPACT=0
exec "$py" -u perf/bcx_p10_duotraj/duo_round.py \
    --rounds "$rounds" --interleave "$duo" --split-gate "$split" --binder 146 \
    --params "$params" --out "$out" "$@"
