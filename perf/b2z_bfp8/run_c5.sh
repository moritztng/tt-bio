#!/usr/bin/env bash
# The whole pass on one whglx card: block screen first, then the 298 aa parity control.
#
# Detach it (`setsid nohup bash perf/b2z_bfp8/run_c5.sh &`) with the cwd INSIDE this worktree, so
# a torn-down parent worktree cannot delete the job's files mid-run.
#
# Card 5 on whglx is a Wormhole chip. Every number it produces is indicative for the Blackhole
# cell and has to be re-confirmed on qb2 before anything lands.
set -u
cd "$(dirname "$0")/../.." || exit 1
WT=$PWD
PY=${PY:-/home/mthuening/work/tt-bio/env/bin/python}
CARD=${CARD:-5}
TAG=${TAG:-whglx_c${CARD}}

run() {
  local name=$1; shift
  echo "=== $name $(date -u +%H:%M:%SZ) ==="
  env TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
      TT_BIO_LEASE_HOLDER=worker:b2z-bfp8-narrow PYTHONPATH="$WT" \
      "$PY" "$@" 2>&1 | grep -vE '^\s*$|DEBUG *\| ttnn|^Config\{'
  echo "=== $name rc=${PIPESTATUS[0]} $(date -u +%H:%M:%SZ) ==="
}

# A bricked chip throws on the first dispatch and a wedged one hangs; either way nothing below is
# worth running, so probe once and bail loudly rather than leaving a 40-minute job to hang.
if ! run probe -c 'from tt_bio import tenstorrent as T; T.get_device(); print("device ok")' \
     | tee /dev/stderr | grep -q "device ok"; then
  echo "card $CARD did not come up; run 'tt-smi -r $CARD' to completion (no external timeout)"
  exit 2
fi

run screen perf/b2z_bfp8/site_screen.py \
    --out "perf/b2z_bfp8/site_screen_512_${TAG}.json" --seq 512 --reps 5
run parity perf/b2z_bfp8/site_parity.py \
    --out "perf/b2z_bfp8/site_parity_298_${TAG}.json" \
    --cifdir "perf/b2z_bfp8/cif_${TAG}"
