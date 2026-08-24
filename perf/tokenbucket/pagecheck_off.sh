#!/usr/bin/env bash
# Leg C attribution arm. pagecheck.sh measured _PAIR_PROJ_BW 16 vs 1 with the token bucket ON in
# both arms (it is the default now) and read 1.203211 A, against a page string of 0.3248 A. That
# number alone cannot say whether the bucket did it: the page string has no in-repo artifact and was
# measured on qb1 / ttnn 0.67.4, while this is qb2 / ttnn 0.68.0. So take the same two arms with the
# bucket forced OFF, which is the page's original condition, and let the 2x2 attribute the delta.
set -u
: "${CARD:?set CARD}"
WT=$(cd "$(dirname "$0")/../.." && pwd)
SLUG=$(basename "$WT")
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
OUT=perf/tokenbucket/pagecheck_off
TGT=perf/size512/fixtures/cdk2x2_298.yaml
A3M=perf/size512/fixtures/cdk2x2_298.a3m
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:$SLUG"

arm() {
  d=$OUT/$1; bw=$2
  if [ -f "$d/r.json" ]; then echo "SKIP $1"; return 0; fi
  mkdir -p "$d"
  echo "=== $(date -Is) BEGIN $1 bw=$bw bucket=OFF"
  env $LEASE TT_BIO_PROTENIX_TOKEN_BUCKET=0 PYTHONPATH=$WT ESM_ROOT=$ESM_ROOT "$PY" -u \
    perf/inblockw/fold_ab.py --model opendde --pair-proj-bw "$bw" --target "$TGT" --a3m "$A3M" \
    --repeat 1 --out "$d/r.json" > "$d/run.log" 2>&1
  echo "=== $(date -Is) END $1 rc=$?"
}

arm 298_bw16off_1 16
arm 298_bw1off_1  1
echo "=== $(date -Is) all off arms attempted"
"$PY" -u perf/other512/cif_rmsd.py "$OUT" --ref bw1off
