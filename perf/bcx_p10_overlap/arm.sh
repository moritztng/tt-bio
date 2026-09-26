#!/bin/bash
# One composed wave-10 round on a qb2 card, with the leg-1 timeline sampler on.
#   arm.sh <tag> <rounds> <on|off> [extra args...]
#
# The round is `bcx-p10-stack2`'s composed arm -- `--triatt-bw 1 --rne-kernel 1 --triatt-hifi 1`
# with TT_BIO_MM_LAYOUT=1 -- driven through `perf/bcx_p10_stack/arm.sh`, which is the qb-native
# launcher. `perf/bcx_p10_stack2/arm.sh` is the pc copy and hardcodes /home/moritz paths.
#
# NINE rounds is the ceiling per process: the BindCraft 2 compile path deadlocks against itself
# at round 10 of this configuration against a cold cache (`bcx-p10-rne`, root-caused in
# `bcx-p10-hostfloor`). Sample size comes from more processes, not longer ones.
#
# `on` here is the OVERLAP arm, not the lever stack: both arms carry all three wave-10 levers
# and the only difference is TT_BIO_BCX_OVERLAP. The lever stack is the round, not the arm.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; mode=$3; shift 3
if [ "$rounds" -gt 9 ]; then echo "arm.sh: 9 rounds is the ceiling, see bcx-p10-rne" >&2; exit 2; fi
case "$mode" in
    on)  export TT_BIO_BCX_OVERLAP=1 ;;
    off) export TT_BIO_BCX_OVERLAP=0 ;;
    *) echo "arm.sh: mode must be on or off, got '$mode'" >&2; exit 2 ;;
esac
export ARM_OUT_ROOT=${ARM_OUT_ROOT:-$PWD/perf/bcx_p10_overlap/out}
export ARM_XLA_CACHE=${ARM_XLA_CACHE:-$PWD/perf/bcx_p10_overlap/out/xlacache}
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-1}
export TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-1}
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-overlap
export TT_BIO_MM_LAYOUT=1
mkdir -p "$ARM_OUT_ROOT"
exec perf/bcx_p10_stack/arm.sh "$tag" "$rounds" 1 1 hifi \
    --triatt-bw 1 --rne-kernel 1 --timeline "${TIMELINE_MS:-2}" "$@"
