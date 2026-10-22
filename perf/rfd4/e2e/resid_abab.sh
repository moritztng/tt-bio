#!/bin/bash
# The residual-stream bet on one card in one session, under the chip flock (D-orch-40): ABAB warm
# designs in one tree at each size, legs named by their env. AICLK is sampled inside every design.
#   perf/rfd4/e2e/resid_abab.sh <card> <out-dir> "<size>:<rounds> ..." "<leg>=<ENV=V,...> ..." [designs per leg]
# e.g. resid_abab.sh 3 artifacts/abab1 "512:2 256:2 128:1" "base=TT_BIO_RFD4_ELIDE_MASKS=0 elide=TT_BIO_RFD4_ELIDE_MASKS=1"
set -u
card=$1 out=$2 sizes=$3 legs=$4 designs=${5:-2}
wt=$(cd "$(dirname "$0")/../../.." && pwd)
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
weights=${RFD4_WEIGHTS:-$HOME/rfd4-bundle}
mkdir -p "$out"
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:$(basename "$wt")
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$card.lock"
stamp "wait flock chip$card"
flock -w 21600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$card" >/dev/null 2>&1; do sleep 20; done
cd "$wt"
stamp "start $(git log --oneline -1)"
for sr in $sizes; do
    n=${sr%%:*} rounds=${sr##*:}
    for r in $(seq 1 "$rounds"); do
        for leg in $legs; do
            name=${leg%%=*} envs=${leg#*=}
            (export ${envs//,/ }
             PYTHONPATH=$wt timeout 2400 "$py" perf/rfd4/e2e/design.py --spec "$HOME/rfd4-inputs/pdl1-$n.yaml" \
                 --weights "$weights" --out "$out/$n-$name$r" --designs "$designs" > "$out/$n-$name$r.log" 2>&1)
            stamp "design $n-$name$r rc=$? $(tail -1 "$out/$n-$name$r/runs.jsonl" 2>/dev/null | cut -c1-400)"
        done
    done
done
stamp end
