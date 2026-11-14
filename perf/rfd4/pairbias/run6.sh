#!/bin/bash
# pairbias run6: the 512 design grade of the A+B stack (candAB), seeds 0-30, rng upstream, to grade
# A+B as a stack against cand (run3+run4, same seeds).
set -u
W=/home/ttuser/.coworker/wt/rfd4-perf-pairbias; O=$W/artifacts/run6; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-pairbias PYTHONPATH=$W
mkdir -p $O/design; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 10800 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start $(git log --oneline -1 | cut -c1-70)"
CAND=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail
leg() {  # leg <name> <levers>
  TT_BIO_LEVERS="$2" timeout 5400 $PY perf/rfd4/e2e/design.py --spec $HOME/rfd4-inputs/pdl1-512.yaml \
      --weights $HOME/rfd4-bundle --out $O/design/$1 --rng upstream --seeds $(seq 0 30) > $O/design/$1.log 2>&1
  stamp "design $1 rc=$? warm=$(grep -c "\"kind\": \"warm\"" $O/design/$1/runs.jsonl 2>/dev/null)"
}
leg candAB $CAND,pair_write_res,pair_bias_rms
stamp end
