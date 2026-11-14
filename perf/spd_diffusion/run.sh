#!/bin/bash
# spd-diffusion on a held chip: one bench process per arm, private lease dir, timeout on each.
# usage: run.sh RUN CHIP COND "arm=ENGINE[:ENV=V,...]" ... ; ENGINE is a tt_bio tree (git archive of a sha).
# SEEDS defaults to 101,102,103,104 (1 cold + 3 warm).
RUN=${1:?run dir}; CHIP=${2:?chip}; COND=${3:?cond.pt}; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
# Host threads at the serving share (2 per chip), as spd-bench measures: an unshared run on a loaded
# Galaxy host times the host, not the chip.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=spd-diffusion TT_BIO_LEASE_DIR=${TT_BIO_LEASE_DIR:-$HOME/spd-diffusion/leases}
mkdir -p "$RUN" "$TT_BIO_LEASE_DIR"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$RUN/run.log"; }
for spec in "$@"; do
  arm=${spec%%=*}; rest=${spec#*=}; eng=${rest%%:*}; envs=""; [[ $rest == *:* ]] && envs=${rest#*:}
  say "arm $arm engine $eng env '$envs' omp $OMP_NUM_THREADS load $(cut -d' ' -f1-3 /proc/loadavg)"
  ( IFS=,; for kv in $envs; do export "$kv"; done
    PYTHONPATH=$eng timeout 5400 ${PY:-python} "$HERE/bench.py" diff "$RUN" "$CHIP" "$COND" "$arm" "${SEEDS:-101,102,103,104}" ${CENSUS:+census} \
      > "$RUN/$arm.log" 2>&1 )
  say "arm $arm rc=$?"
done
say "end"
