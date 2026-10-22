#!/bin/bash
# Bet 11 on the trunk pass alone: pass_bench.py per arm, interleaved, under the chip flock. The traced
# pass time is device-bound, so qb2 host load does not move it the way it moves a design wall.
#   perf/rfd4/e2e/resid_pass_abab.sh <card> <out-dir> "<size> ..." <rounds>
set -u
card=$1 out=$2 sizes=$3 rounds=$4
wt=$(cd "$(dirname "$0")/../../.." && pwd)
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
mkdir -p "$out"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$card.lock"
stamp "wait flock chip$card"
flock -w 21600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$card" >/dev/null 2>&1; do sleep 20; done
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:$(basename "$wt") TT_BIO_TRACE_REGION_MB=1536
cd "$wt"
stamp "start $(git log --oneline -1)"
for n in $sizes; do
    for r in $(seq 1 "$rounds"); do
        for leg in base=TT_BIO_RFD4_ELIDE_MASKS=0 elide=TT_BIO_RFD4_ELIDE_MASKS=1 \
                   pairbf16=TT_BIO_RFD4_ELIDE_MASKS=1,TT_BIO_LEVERS=trimul_t1+pair_bf16; do
            name=${leg%%=*} envs=${leg#*=}
            (export ${envs//,/ }
             PYTHONPATH=$wt timeout 1800 "$py" perf/rfd4/e2e/pass_bench.py --weights "${RFD4_WEIGHTS:-$HOME/rfd4-bundle}" \
                 --n "$n" --reps 10 --out "$out/$n-$name$r.json" > "$out/$n-$name$r.log" 2>&1)
            rc=$?
            stamp "pass $n-$name$r rc=$rc $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[\"ops_per_pass\"], d[\"eager_ms\"], d.get(\"traced_ms\"), d.get(\"call_traced_ms\"), d[\"aiclk\"])" "$out/$n-$name$r.json" 2>/dev/null)"
        done
    done
done
stamp end
