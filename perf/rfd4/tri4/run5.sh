#!/bin/bash
# tri4 run5: the stack across sizes (stack_ab.py, paired, 30 reps): it must serve or decline cleanly at every bucket
# T1 takes (>= 160 tokens), and this is its speed there.
#   perf/rfd4/tri4/run5.sh <card>
set -u
W=/home/ttuser/.coworker/wt/rfd4-perf-tri4; O=$W/artifacts/tri4/run5; PY=$HOME/tt-bio-dev/env/bin/python
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
for n in 128 160 224 320 384 512 640; do
  timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle --out $O/$n-stack.json --n $n --reps 30 \
      --base $B --levers $S > $O/$n-stack.log 2>&1
  stamp "ab $n-stack rc=$? $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[\"z_rel_rms_stack_vs_base\"], d[\"base\"][\"trunk_s_per_design\"], d[\"stack\"][\"trunk_s_per_design\"], d[\"paired_speedup_median\"], d[\"aiclk\"])" $O/$n-stack.json 2>/dev/null)"
done
stamp end
