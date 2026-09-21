set -u
W=/home/ttuser/.coworker/wt/allm-model
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_model/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=1
cd "$W" || exit 1
run() {
  echo "=== $(date -u +%H:%M:%SZ) START $*"
  BENCHLOCK_WAIT_S=1500 BENCHLOCK_LOAD_WAIT_S=150 "$BL" allm-model -- \
    env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-model \
        PYTHONPATH="$W" TT_BIO_AICLK=1350 \
    "$PY" -u "$W/perf/allm_model/blockcensus.py" --card $CARD "$@" 2>&1 \
    | stdbuf -oL grep -vE "^Loading|^Fetching|it/s\]$|DEBUG|^Config\{|^Warning:|^ *$"
  echo "=== $(date -u +%H:%M:%SZ) END rc=${PIPESTATUS[0]} $*"
}
# ESMFold2 with the lever counters: does the pair-FFN row-block family fire on its 532
# transition calls, or is it declined on 100 % of them the way E6 was on boltz2's trimuls?
run --model esmfold2 --size 512 --folds "cold,A,B:3,B" --out "$OUT/esmfold2_512_levers.json"
# Boltz-2 at depth 4: TrunkModule > Pairformer > PairformerLayer > TriangleMultiplication.
run --model boltz2 --size 512 --folds "cold,A,B:4,B,B:4,C" --out "$OUT/b2census_512_d4.json"
echo "ALLDONE-chain6"
