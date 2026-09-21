#!/usr/bin/env bash
# The structural delta on a cell where the seed floor is a floor.
#
# The first attempt ran the digest harness's cell -- 128 tokens, --single_sequence, 6 sampling
# steps -- and two seeds landed 106 A (openfold3) and 152 A (protenix-v2) apart. A floor that
# size passes anything, so it measures nothing. This is the 512-token cell with its MSA and the
# model's own default sampling steps, which is the cell the 0.60 A kill bar and the 1.84 A seed
# floor are quoted for.
set -u
WT=/home/ttuser/.coworker/wt/of3t-fwdkcfg
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT="$WT/perf/of3t_fwdkcfg"
WORK=$(mktemp -d /tmp/of3t-fwdkcfg-p3-XXXXXX)
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-fwdkcfg
"$PY" perf/of3t_fwdkcfg/structure_delta.py \
  --python "$PY" --fixture "$WT/perf/size512/fixtures/cdk2x2_512.yaml" --card 0 \
  --workdir "$WORK" --out "$OUT/STRUCTURE_DELTA_512_qb2c0.json" \
  --case openfold3=openfold3.diffusion_transformer \
  --case protenix-v2=protenix.token_dit
echo "=== exit=$? $(date -u +%FT%TZ) ==="
rm -rf "$WORK"
echo "PHASE3 DONE $(date -u +%FT%TZ)"
