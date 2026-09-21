set -u
W=/home/ttuser/.coworker/wt/allm-model
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_model/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=1
cd "$W" || exit 1
run() {
  echo "=== $(date -u +%H:%M:%SZ) START $*"
  BENCHLOCK_WAIT_S=900 BENCHLOCK_LOAD_WAIT_S=90 "$BL" allm-model -- \
    env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-model \
        PYTHONPATH="$W" TT_BIO_AICLK=1350 \
    "$PY" -u "$W/perf/allm_model/blockcensus.py" --card $CARD "$@" 2>&1 \
    | stdbuf -oL grep -vE "^Loading|^Fetching|it/s\]$|DEBUG|^Config\{|^Warning:|Cached data not found|^ *$|Environment variable|^  \(|^Using element|DeprecationWarning|self.pid|parsed = parse"
  echo "=== $(date -u +%H:%M:%SZ) END rc=${PIPESTATUS[0]} $*"
}
run --model esmfold2 --size 512 --folds "cold,A,B:3" --out "$OUT/esmfold2_512_levers.json"
run --model rf3 --size 512 --folds "cold,A,K,B:1,B" --out "$OUT/rf3_512.json"
run --model boltz2 --size 512 --folds "cold,A,B:4,B" --out "$OUT/b2census_512_d4.json"
echo "ALLDONE-chain7"
