#!/bin/bash
# tri4 run4: the stack grade, pdl1 at 512, seeds 0-9, rng upstream: main's default set, + psig + hifi2 (ph2),
# + trimul_t1w8 (ph2w8). Graded as stacks against base (CA RMSD, seed floor) by perf/rfd4/e2e/grade.py.
#   perf/rfd4/tri4/run4.sh <card>
set -u
W=/home/ttuser/.coworker/wt/rfd4-perf-tri4; O=$W/artifacts/tri4/run4; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri4 PYTHONPATH=$W:$W/perf
mkdir -p $O/design; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 10800 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start $(git log --oneline -1 | cut -c1-70) dirty=$(git status --short tt_bio | wc -l)"
B=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail,trimul_t1_dram,pair_bias_rms,pair_write_res
leg() {  # leg <name> <levers>
  TT_BIO_LEVERS="$2" timeout 5400 $PY perf/rfd4/e2e/design.py --spec $HOME/rfd4-inputs/pdl1-512.yaml \
      --weights $HOME/rfd4-bundle --out $O/design/$1 --rng upstream --seeds $(seq 0 9) > $O/design/$1.log 2>&1
  stamp "design $1 rc=$? warm=$(grep -c "\"kind\": \"warm\"" $O/design/$1/runs.jsonl 2>/dev/null)"
}
leg base $B
leg ph2w8 $B,trimul_t1_psig,trimul_gin_hifi2,trimul_t1w8
leg ph2 $B,trimul_t1_psig,trimul_gin_hifi2
stamp end
