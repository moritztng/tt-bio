set -u
# Boltz-2 at depth 4. Its trimul sits TrunkModule > Pairformer > PairformerLayer >
# TriangleMultiplication, so B:2 stops two levels above it and only the K count reaches it.
W=/home/ttuser/.coworker/wt/allm-model
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_model/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=1
cd "$W" || exit 1
echo "=== $(date -u +%H:%M:%SZ) START boltz2 B:4"
BENCHLOCK_WAIT_S=1500 BENCHLOCK_LOAD_WAIT_S=180 "$BL" allm-model -- \
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-model \
      PYTHONPATH="$W" TT_BIO_AICLK=1350 \
  "$PY" -u "$W/perf/allm_model/blockcensus.py" --card $CARD --model boltz2 --size 512 \
    --folds "cold,A,B:4,B,B:4,C" --out "$OUT/b2census_512_d4.json" 2>&1 \
  | stdbuf -oL grep -vE "^Loading|^Fetching|it/s\]$|DEBUG|^Config\{|^Warning:|^ *$"
echo "=== $(date -u +%H:%M:%SZ) END rc=${PIPESTATUS[0]} boltz2 B:4"
echo "ALLDONE-chain6"
