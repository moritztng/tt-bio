set -u
# RoseTTAFold3: walk the tape into model.Recycler (50.8 % of the fold at depth 1) so the fold
# splits into SHARED classes and rf3's own code -- the question the brief asks and the one a
# depth-1 census cannot answer.
W=/home/ttuser/.coworker/wt/allm-model
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_model/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=1
cd "$W" || exit 1
echo "=== $(date -u +%H:%M:%SZ) START rf3 ladder"
BENCHLOCK_WAIT_S=1500 BENCHLOCK_LOAD_WAIT_S=120 "$BL" allm-model -- \
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-model \
      PYTHONPATH="$W" TT_BIO_AICLK=1350 \
  "$PY" -u "$W/perf/allm_model/blockcensus.py" --card $CARD --model rf3 --size 512 \
    --folds "cold,A,B:2,B:3,B:4,B" --out "$OUT/rf3_512_ladder.json" 2>&1 \
  | stdbuf -oL grep -vE "^Loading|^Fetching|it/s\]$|DEBUG|^Config\{|^Warning:|Cached data not found|^ *$|Environment variable|^  \(|^Using element|DeprecationWarning|self.pid|parsed = parse"
echo "=== $(date -u +%H:%M:%SZ) END rc=${PIPESTATUS[0]} rf3 ladder"
echo "ALLDONE-chain8"
