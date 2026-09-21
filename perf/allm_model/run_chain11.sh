set -u
# BoltzGen membership, COUNTED. The driver drives the CLI, so every design is cold and there is
# no warm A/A floor -- which is why this asks for a count (K) and not for shares.
W=/home/ttuser/.coworker/wt/allm-model
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_model/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=1
cd "$W" || exit 1
echo "=== $(date -u +%H:%M:%SZ) START boltzgen K"
BENCHLOCK_WAIT_S=1200 BENCHLOCK_LOAD_WAIT_S=90 "$BL" allm-model -- \
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-model \
      PYTHONPATH="$W" TT_BIO_AICLK=1350 \
  "$PY" -u "$W/perf/allm_model/blockcensus.py" --card $CARD --model boltzgen --driver boltzgen \
    --steps 40 --folds "cold,K" --out "$OUT/boltzgen_membership.json" 2>&1 \
  | stdbuf -oL grep -vE "^Loading|^Fetching|it/s\]$|DEBUG|^Config\{|^Warning:|^ *$|Cached data not found"
echo "=== $(date -u +%H:%M:%SZ) END rc=${PIPESTATUS[0]} boltzgen"
echo "ALLDONE-chain11"
