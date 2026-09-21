set -u
W=/home/ttuser/.coworker/wt/allm-model
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_model/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=1
cd "$W" || exit 1
run() {
  echo "=== $(date -u +%H:%M:%SZ) START $*"
  BENCHLOCK_WAIT_S=900 BENCHLOCK_LOAD_WAIT_S=120 "$BL" allm-model -- \
    env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-model \
        PYTHONPATH="$W" TT_BIO_AICLK=1350 \
    "$PY" -u "$W/perf/allm_model/blockcensus.py" --card $CARD "$@" 2>&1 \
    | stdbuf -oL grep -vE "^Loading|^Fetching|it/s\]$|DEBUG|^Config\{|^Warning:|^ *$"
  echo "=== $(date -u +%H:%M:%SZ) END rc=${PIPESTATUS[0]} $*"
}
run --model boltz2 --size 512 --folds "cold,A,K,B:2,B" --out "$OUT/b2census_512.json"
run --model rf3    --size 512 --folds "cold,A,K,B:1,B:2,B" --out "$OUT/rf3_512.json"
run --model rfd3 --driver rfd3 --folds "cold,A,K,B:3,B" --out "$OUT/rfd3_membership.json"
echo "ALLDONE-chain3"
