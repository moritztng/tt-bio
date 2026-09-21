set -u
W=/home/ttuser/.coworker/wt/allm-model
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_model/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=1
cd "$W" || exit 1
run() {
  echo "=== $(date -u +%H:%M:%SZ) START $*"
  BENCHLOCK_WAIT_S=1500 BENCHLOCK_LOAD_WAIT_S=120 "$BL" allm-model -- \
    env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-model \
        PYTHONPATH="$W" TT_BIO_AICLK=1350 \
    "$PY" -u "$W/perf/allm_model/blockcensus.py" --card $CARD "$@" 2>&1 \
    | stdbuf -oL grep -vE "^Loading|^Fetching|it/s\]$|DEBUG|^Config\{|^Warning:|Cached data not found|^ *$|Environment variable|^  \(|^Using element|DeprecationWarning|self.pid|parsed = parse"
  echo "=== $(date -u +%H:%M:%SZ) END rc=${PIPESTATUS[0]} $*"
}
# 2 folds only: all I need is TRIMUL_GOUT_REJECTS, which names the blocking clause.
run --model esmfold2 --size 512 --folds "cold,A" --out "$OUT/esmfold2_512_rejects.json"
# RFdiffusion3 on its PUBLISHED cell: rfd3_R4.json is contig A1-585,100 = 685 residues, and 200
# timesteps is the production rollout. Pass 1's capture was 120 residues at 4 steps.
run --model rfd3 --driver rfd3 --target "$W/perf/dsfix/fixtures/rfd3_R4.json" --steps 200 \
    --folds "cold,A,K,B:2,B" --out "$OUT/rfd3_R4_685.json"
echo "ALLDONE-chain9"
