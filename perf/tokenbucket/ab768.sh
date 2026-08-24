#!/usr/bin/env bash
# The one thing standing between this branch and a verdict: is the size ladder's +9.1 % at 768 aa
# the token-bucket pad, or the co-tenant that was on the box when that rung ran?
#
# The ladder cannot answer it. Its two arms are separate processes minutes apart, and its 768 rung
# was the ON arm's LAST rung, measured as load climbed from 1.50 to 6.30, while the OFF arm's 768
# rung ran quiet. That is a load offset sitting exactly on the number in question. The lever deltas
# do not predict a regression there either: 768 shows the TRIMUL_IN_PROJ_DUAL_NOC win and loses no
# L1 lever, while 640 -- the rung that DOES lose TRIATT_PERSISTENT_MASK -- got faster.
#
# So: the paired instrument, at 768. One process, both arms, alternating within-pair order, so a
# load excursion lands on both arms of a pair and largely cancels in the difference. 3 pairs is
# enough: the effect under test is ~17 s against a within-arm spread of ~1.4 s.
#
# 768 aa costs ~195 s a fold (the ladder read 183/199 s, tt_baseline read 83 s at 512 where the
# ladder read 79 s, so the two protocols price alike). 3 pairs plus 2 cold folds is ~26 min.
set -u
: "${CARD:?set CARD to this launch grant}"
WT=$(cd "$(dirname "$0")/../.." && pwd)
SLUG=${SLUG:-$(basename "$WT")}
PAIRS=${PAIRS:-3}
PY=${GATE_PYTHON:-/home/ttuser/.coworker/rel070/relvenv/bin/python3}
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
export PATH=/home/ttuser/.local/bin:/home/ttuser/tt-bio/env/bin:$PATH
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:$SLUG"
OUT=perf/tokenbucket/ab768
mkdir -p "$OUT"

# A dead card shows as 100 % of one core, not 0 %, and cost a 42-minute fold into nothing on
# 2026-08-23. A 256x256 matmul probe is the only thing that tells wedged from slow.
echo "$(date -Is) preflight card $CARD"
if ! timeout 300 env $LEASE PYTHONPATH=$WT "$PY" -u perf/tokenbucket/preflight_card.py > "$OUT/preflight.log" 2>&1; then
  echo "$(date -Is) PREFLIGHT FAILED on card $CARD. Not folding."; exit 1
fi
echo "$(date -Is) preflight OK, load $(cut -d' ' -f1-3 /proc/loadavg)"

env $LEASE PYTHONPATH=$WT ESM_ROOT=$ESM_ROOT \
  "$PY" -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat "$PAIRS" \
    --ab-env TT_BIO_PROTENIX_TOKEN_BUCKET --ab-values 1,0 \
    --target perf/size512/fixtures/cdk2x2_768.yaml \
    --msa-a3m perf/size512/fixtures/cdk2x2_768.a3m \
    --msa-dir "$WT/.msa_tokenbucket" --keep-cif "$OUT/cif" \
    --label "od768 $PAIRS pairs, page protocol, qb2 card $CARD, paired token-bucket A/B" \
    --out "$OUT/od768_paired.json" > "$OUT/run.log" 2>&1
echo "=== $(date -Is) od768 END rc=$? load $(cut -d' ' -f1-3 /proc/loadavg)"
tail -25 "$OUT/run.log"
