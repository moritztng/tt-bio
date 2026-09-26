#!/bin/bash
# The round A/B for the cheap generic_op dispatch: `--rne-kernel 1` on BOTH arms, so there are
# ~1900 dispatches a round to make cheaper, and TT_BIO_GENQ_COMPACT the only difference.
#
#   ab.sh [rounds] [tag-suffix] [arms...]
#
# NINE rounds a process is the ceiling and it is not a sample-size choice: BindCraft 2's compile
# path deadlocks against itself at round 10 of this configuration against a cold cache, every
# time, on the same program hash (`state/perf10/bcx-p10-rne.md`). Round 1 is the compile and is
# dropped, so a process gives 8 timed rounds.
#
# The arms alternate at the process boundary and the default order is symmetric, so a monotone
# drift over the sitting cancels. A sitting split across two launches runs `off on` then `on off`.
set -euo pipefail
cd "$(dirname "$0")/../.."
rounds=${1:-9}
suffix=${2:-$(date +%H%M%S)}
shift 2 || true
arms=("${@:-off on on off}")
[ $# -eq 0 ] && arms=(off on on off)
export ARM_OUT_ROOT=perf/bcx_p10_genq/out
export ARM_XLA_CACHE=$PWD/perf/bcx_p10_genq/out/xlacache
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-genq
# pc, not qb: this row's claim is host microseconds, so it is the one row of wave 11 that cannot
# move to a quieter box with a different host.
export ARM_PYTHON=/home/moritz/bcx_hostcut_venv/bin/python
export ARM_PYTHONPATH=/home/moritz/bcx_shipped/bc2
mkdir -p perf/bcx_p10_genq/out
for arm in "${arms[@]}"; do
    tag=p_${arm}_${suffix}_$RANDOM
    echo "=== $tag  rounds=$rounds  genq-compact=$arm  $(date -u +%FT%TZ)" >&2
    TT_BIO_GENQ_COMPACT="$([ "$arm" = on ] && echo 1 || echo 0)" \
    bash perf/bcx_p10_stack/arm.sh "$tag" "$rounds" 1 1 hifi --rne-kernel 1 \
        > "perf/bcx_p10_genq/out/$tag.log" 2>&1
    echo "=== $tag done $(date -u +%FT%TZ)" >&2
done
