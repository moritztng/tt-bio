#!/usr/bin/env bash
# The structural delta on a cell where the seed floor is a floor.
#
# The first attempt ran the digest harness cell -- 128 tokens, --single_sequence, 6 sampling
# steps -- and two seeds landed 106 A (openfold3) and 152 A (protenix-v2) apart under Kabsch
# superposition. A floor that size passes anything, so it measures nothing. This is the
# 512-token cell with its MSA and the model own default sampling steps, which is the cell the
# 0.60 A kill bar and the 1.84 A seed floor are quoted for.
#
# HOST is named in the artifact: qb1 card 0 is a p150a, qb2 card 0 a p300c. Never pc card 0.
set -u
WT=/home/ttuser/.coworker/wt/of3t-fwdkcfg
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT="$WT/perf/of3t_fwdkcfg"
TAG=${TAG:-qb1c0}
BOARD=${BOARD:-p150a}
WORK=$(mktemp -d "$WT/.tmp-p3-XXXXXX")
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-fwdkcfg
"$PY" perf/of3t_fwdkcfg/structure_delta.py \
  --python "$PY" --fixture "$WT/perf/size512/fixtures/cdk2x2_512.yaml" --card 0 \
  --board "$BOARD" \
  --workdir "$WORK" --out "$OUT/STRUCTURE_DELTA_512_${TAG}.json" \
  --case openfold3=openfold3.diffusion_transformer \
  --case protenix-v2=protenix.token_dit
echo "=== exit=$? $(date -u +%FT%TZ) ==="
rm -rf "$WORK"
echo "PHASE3 FINISHED $(date -u +%FT%TZ)"
