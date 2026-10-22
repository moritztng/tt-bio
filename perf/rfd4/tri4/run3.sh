#!/bin/bash
# tri4 run3: the levers in the trunk, paired in one process (perf/rfd4/bfp8/stack_ab.py: two trunks, traced passes
# replayed A B A B, z compared), 30 reps, against main's default set. Then the same at 256 for the stack.
#   perf/rfd4/tri4/run3.sh <card>
set -u
W=/home/ttuser/.coworker/wt/rfd4-perf-tri4; O=$W/artifacts/tri4/run3; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri4 PYTHONPATH=$W:$W/perf
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 10800 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start $(git log --oneline -1 | cut -c1-70) dirty=$(git status --short tt_bio | wc -l)"
B=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail,trimul_t1_dram,pair_bias_rms,pair_write_res
S=$B,trimul_t1_psig,trimul_gin_hifi2,trimul_t1w8
ab() {  # ab <n> <name> <levers>
  timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle --out $O/$1-$2.json --n $1 --reps 30 \
      --base $B --levers $3 > $O/$1-$2.log 2>&1
  stamp "ab $1-$2 rc=$? $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: v for k, v in d.items() if k not in (\"base\", \"levers\")})" $O/$1-$2.json 2>/dev/null | cut -c1-600)"
}
# ab 512 cm (pair_tail_cm, reverted in f0e6db866)
ab 512 psig $B,trimul_t1_psig
ab 512 ph2 $B,trimul_t1_psig,trimul_gin_hifi2
ab 512 ph2w8 $B,trimul_t1_psig,trimul_gin_hifi2,trimul_t1w8
ab 512 stack $S
ab 256 stack $S
stamp end
