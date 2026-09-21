set -u
# allm-model: RFdiffusion3 on its PUBLISHED cell. perf/dsfix/fixtures/rfd3_R4.json is contig
# A1-585,100 = 585 target + 100 designed = 685 residues, which is the size the perf page's
# 92.239 s/design cell was taken at; 200 timesteps is the production rollout. Pass 1's capture
# was a 120-residue contig at 4 timesteps and its shares are not production shares.
W=/home/ttuser/.coworker/wt/allm-model
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_model/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=1
cd "$W" || exit 1
echo "=== $(date -u +%H:%M:%SZ) START rfd3 R4 685aa 200 steps"
BENCHLOCK_WAIT_S=1500 BENCHLOCK_LOAD_WAIT_S=120 "$BL" allm-model -- \
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-model \
      PYTHONPATH="$W" TT_BIO_AICLK=1350 \
  "$PY" -u "$W/perf/allm_model/blockcensus.py" --card $CARD --model rfd3 --driver rfd3 \
    --target "$W/perf/dsfix/fixtures/rfd3_R4.json" --steps 200 \
    --folds "cold,A,K,B:2,B" --out "$OUT/rfd3_R4_685.json" 2>&1 \
  | stdbuf -oL grep -vE "^Loading|^Fetching|it/s\]$|DEBUG|^Config\{|^Warning:|^ *$"
echo "=== $(date -u +%H:%M:%SZ) END rc=${PIPESTATUS[0]} rfd3 R4"
echo "ALLDONE-chain5"
