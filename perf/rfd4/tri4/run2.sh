#!/bin/bash
# tri4 run2: the T1 levers in the trunk pass at 512 (pass_bench.py, traced, 30 reps), arms interleaved over
# rounds under the chip flock. Round 1 dumps the pass output for the z comparison against base.
#   perf/rfd4/tri4/run2.sh <card> [rounds] [n]
set -u
W=/home/ttuser/.coworker/wt/rfd4-perf-tri4; O=$W/artifacts/tri4/run2; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; R=${2:-3}; N=${3:-512}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri4 PYTHONPATH=$W:$W/perf \
       TT_BIO_TRACE_REGION_MB=1536
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 10800 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start $(git log --oneline -1 | cut -c1-70)"
B=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail,trimul_t1_dram,pair_bias_rms,pair_write_res
for r in $(seq 1 $R); do
  for leg in base=$B psig=$B,trimul_t1_psig ph2=$B,trimul_t1_psig,trimul_gin_hifi2 \
             ph2w8=$B,trimul_t1_psig,trimul_gin_hifi2,trimul_t1w8; do
    name=${leg%%=*} lev=${leg#*=}
    dump=(); [ $r = 1 ] && dump=(--dump $O/$N-$name.pt)
    TT_BIO_LEVERS=$lev timeout 1800 $PY perf/rfd4/e2e/pass_bench.py --weights $HOME/rfd4-bundle --n $N --reps 30 \
        --out $O/$N-$name$r.json "${dump[@]}" > $O/$N-$name$r.log 2>&1
    rc=$?
    stamp "pass $N-$name$r rc=$rc $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(\"traced_ms\"), d[\"eager_ms\"], d[\"aiclk\"])" $O/$N-$name$r.json 2>/dev/null)"
  done
done
stamp end
