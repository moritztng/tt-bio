#!/usr/bin/env bash
# The three gated sizes, one after another, on one chip: run.py per size against a checkout of main.
#
#   perf/rfd4/fair/fair.sh TREE CARD PART WEIGHTS INPUTS OUT [SIZES] [PY]
#
# TREE is a checkout of tt-bio main (it is what gets measured; this harness only drives it), CARD the
# leased chip, PART p150a for the gated number, INPUTS the directory holding the frozen
# pdl1-{128,256,512}.yaml and their INDEX.json. A p150a run refuses a tree that is dirty or not at
# origin/main, and a spec whose sha256 is not the frozen one. Detach it; it writes OUT/fair.log.
set -u
TREE=${1:?main checkout}; CARD=${2:?card}; PART=${3:?part}; WEIGHTS=${4:?weights}; INPUTS=${5:?inputs dir}
OUT=${6:?out dir}; SIZES=${7:-128 256 512}; PY=${8:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
exec >>"$OUT/fair.log" 2>&1
echo "start $(date -u +%FT%TZ) pid=$$ card=$CARD part=$PART tree=$TREE"

git -C "$TREE" fetch -q origin main
SHA=$(git -C "$TREE" rev-parse HEAD)
if [ "$PART" = p150a ]; then
    [ "$SHA" = "$(git -C "$TREE" rev-parse origin/main)" ] || { echo "REFUSE: $TREE is at $SHA, not origin/main"; exit 3; }
    [ -z "$(git -C "$TREE" status --porcelain --untracked-files=no)" ] || { echo "REFUSE: $TREE is dirty"; exit 3; }
fi
for s in $SIZES; do
    want=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['files'][sys.argv[2]]['sha256'])" \
        "$INPUTS/INDEX.json" "pdl1-$s.yaml") || exit 3
    [ "$(sha256sum < "$INPUTS/pdl1-$s.yaml" | cut -d' ' -f1)" = "$want" ] || { echo "REFUSE: pdl1-$s.yaml is not the frozen spec"; exit 3; }
done

export PYTHONPATH="$TREE" TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" TT_BIO_LEASE_HOLDER=worker:rfd4-fair-tt
uptime
for s in $SIZES; do
    echo "size $s start $(date -u +%FT%TZ)"
    timeout 14400 "$PY" "$HERE/run.py" --spec "$INPUTS/pdl1-$s.yaml" --weights "$WEIGHTS" --out "$OUT/$s" \
        --part "$PART" | grep -E '"kind": "(cold|climb|summary|error|abort)"|Error'
    echo "size $s rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
done
echo "end $(date -u +%FT%TZ)"
