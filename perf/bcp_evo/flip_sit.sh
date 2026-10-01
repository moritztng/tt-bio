#!/bin/bash
# A lever read round by round inside ONE serial (N=1) process: duo_round.py --flip sets it per
# round, ABBA from round 3, so the host load that swamps a process-boundary sitting on a shared
# qb2 is common-mode across each quad. Repeated over P processes.
#   flip_sit.sh <card> <out subdir> <module:ATTR> <processes> <rounds> [VAR=value...]
set -uo pipefail
cd "$(dirname "$0")/../.."
c=$1; o=$2; f=$3; p=$4; r=$5; shift 5
export TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c TT_BIO_LEASE_HOLDER=worker:bcp-evo
export OUT_DIR=perf/bcp_evo/out/$o BCP_DUO=perf/bcx_p10_duotraj/duo_round.py
mkdir -p "$OUT_DIR"
for i in $(seq 1 "$p"); do
    echo "=== flip$i $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"
    # round_arm.sh resolves N by auto; pass the serial arm explicitly after "--".
    perf/bcp_evo/round_arm.sh "flip$i" "$r" "$@" -- --interleave 0 --trajectories 1 --flip "$f" \
        > "$OUT_DIR/flip$i.log" 2>&1 || echo "  flip$i exited $?"
done
~/bcx_e2e_venv/bin/python3 perf/bcp_device/split.py $(for i in $(seq 1 "$p"); do echo "$OUT_DIR/flip$i"; done) > "$OUT_DIR/split.txt" 2>&1
echo "=== done $(date -u +%FT%TZ)"
