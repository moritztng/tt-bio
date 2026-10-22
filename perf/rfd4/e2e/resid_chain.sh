#!/bin/bash
# One card session for bet 11: the bf16-pair float64 grade, then the ABAB at 512, 256 and 128.
#   perf/rfd4/e2e/resid_chain.sh <card> <out-dir>
set -u
card=$1 out=$2
wt=$(cd "$(dirname "$0")/../../.." && pwd)
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
mkdir -p "$out"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$card.lock"
stamp "wait flock chip$card"
flock -w 21600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$card" >/dev/null 2>&1; do sleep 20; done
export RESID_FLOCK_HELD=1 TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:$(basename "$wt")
cd "$wt"
PYTHONPATH=$wt timeout 1800 "$py" perf/rfd4/e2e/resid_precision.py --out "$out/precision.json" > "$out/precision.log" 2>&1
stamp "precision rc=$?"
bash perf/rfd4/e2e/resid_abab.sh "$card" "$out/abab" "512:2 256:2 128:1" \
    "base=TT_BIO_RFD4_ELIDE_MASKS=0 elide=TT_BIO_RFD4_ELIDE_MASKS=1 pairbf16=TT_BIO_RFD4_ELIDE_MASKS=1,TT_BIO_RFD4_PAIR_BF16=1"
stamp "chain end"
