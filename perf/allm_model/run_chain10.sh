set -u
# OpenDDE re-censused on ONE card under benchlock. Pass 1's capture fanned three models across
# three cards of this box at once, so its A/A was 2.33 % and its tape charge 4.01 %. M20 needs
# OpenDDE's trunk share against a Boltz-2 arm on the same tree, board and clock; the Boltz-2 arm
# exists (b2census_512_d4.json, same tree, same card, same held clock).
W=/home/ttuser/.coworker/wt/allm-model
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_model/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=1
cd "$W" || exit 1
echo "=== $(date -u +%H:%M:%SZ) START opendde"
BENCHLOCK_WAIT_S=1500 BENCHLOCK_LOAD_WAIT_S=240 "$BL" allm-model -- \
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-model \
      PYTHONPATH="$W" TT_BIO_AICLK=1350 \
  "$PY" -u "$W/perf/allm_model/blockcensus.py" --card $CARD --model opendde --size 512 \
    --folds "cold,A,K,B:2,B:4,B" --out "$OUT/opendde_512_v2.json" 2>&1 \
  | stdbuf -oL grep -vE "^Loading|^Fetching|it/s\]$|DEBUG|^Config\{|^Warning:|^ *$"
echo "=== $(date -u +%H:%M:%SZ) END rc=${PIPESTATUS[0]} opendde"
echo "ALLDONE-chain10"
