#!/usr/bin/env bash
# The clocked repeat of the wide-k trunk-stage arm. Benchlocked because qb1 carries sibling PVX
# rows and a co-tenanted timed A/B is a wrong measurement, not a slow one: if the box never goes
# quiet benchlock exits 75 and nothing is written, which is the intended outcome.
set -u
cd /home/ttuser/.coworker/wt/pvx-land || exit 1
BENCHLOCK_WAIT_S=${BENCHLOCK_WAIT_S:-900} BENCHLOCK_LOAD_WAIT_S=${BENCHLOCK_LOAD_WAIT_S:-900} \
  ~/.coworker/scripts/benchlock.sh pvx-land -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/pvx_land/widek_stage_clocked.py \
    --card 0 --node 0 --mhz 1350 --order off,on,off,on --seed 0 --steps 200 \
    --msa-dir /home/ttuser/widek_msa \
    --workdir /home/ttuser/.coworker/wt/pvx-land/perf/pvx_land/stage_clocked_work \
    --out perf/pvx_land/widek_stage_clocked.json
echo "=== driver exit $? at $(date -Is) ==="
