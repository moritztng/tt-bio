#!/usr/bin/env bash
# Leg C of the token-bucket rebase: re-measure the OpenDDE perf-page parity string at the new
# default. The page says "0.3248 A CA RMSD" for the tuned `in0_block_w` pair-projection config,
# measured on a 298 aa monomer the token bucket now pads to 320, so the number has to be taken
# again with the bucket on (it is the default now) in both arms.
#
# Three folds, one process each, same card: bw16 twice (the A/A floor -- on Blackhole this is a
# noise floor, not necessarily 0) and bw1 once (the byte-identical arm). Scored on RMSD only;
# the host is carrying two other workers, so the timings this writes are not campaign numbers.
set -u
: "${CARD:?set CARD}"
WT=$(cd "$(dirname "$0")/../.." && pwd)
SLUG=$(basename "$WT")
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
OUT=perf/tokenbucket/pagecheck
TGT=perf/size512/fixtures/cdk2x2_298.yaml
A3M=perf/size512/fixtures/cdk2x2_298.a3m
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:$SLUG"

arm() {  # arm <dirname> <bw>
  d=$OUT/$1; bw=$2
  if [ -f "$d/r.json" ]; then echo "SKIP $1 (already on disk)"; return 0; fi
  mkdir -p "$d"
  echo "=== $(date -Is) BEGIN $1 bw=$bw"
  env $LEASE PYTHONPATH=$WT ESM_ROOT=$ESM_ROOT "$PY" -u perf/inblockw/fold_ab.py \
    --model opendde --pair-proj-bw "$bw" --target "$TGT" --a3m "$A3M" --repeat 1 \
    --out "$d/r.json" > "$d/run.log" 2>&1
  echo "=== $(date -Is) END $1 rc=$?"
}

arm 298_bw16_1 16
arm 298_bw16_2 16
arm 298_bw1_1  1
echo "=== $(date -Is) all arms attempted"
"$PY" -u perf/other512/cif_rmsd.py "$OUT" --ref bw1
