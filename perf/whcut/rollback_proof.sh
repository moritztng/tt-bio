#!/bin/bash
# Does the config-only rollback actually restore pre-cutover numerics?
#
# §7.4/§97 offer a ~3 minute rollback that sets TT_BIO_SDPA_DIV_K=0 and
# TT_BIO_SEQ_LEN_MORE_CHUNKING=608 in a systemd drop-in, claiming it "returns Boltz-2's
# numerics and chunking to pre-cutover behaviour while leaving every ring-1 fix in place".
# Nobody has tested that claim, and a rollback nobody has exercised is a hope.
#
# It is directly checkable: fold 640 aa on the ASSEMBLED tree with both switches set to
# their pre-cutover values, and compare the CIF digest against the pre-cutover tree's own
# 640 aa fold. §6.3 measured whpre 640 = 07d7a43010403b18 and whcut 640 = 41c6edf98868d66b,
# and §109 showed the trees are byte-identical below K3's band, so if the switches work the
# reverted arm must land on whpre's digest exactly.
set -u
TREE=/home/cust-team/mthuening/whbase/wt-whcut
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python3
OUT=$TREE/perf/whcut/out/wh/rollback
CARD=${CARD:-28}
mkdir -p "$OUT"
cd "$TREE" || exit 1
echo "ROLLBACK PROOF START $(date -u +%FT%TZ)"

tmp="$OUT/tmp_revert"; rm -rf "$tmp"; mkdir -p "$tmp"
env TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL TMPDIR="$tmp" \
    TT_BIO_SDPA_DIV_K=0 TT_BIO_SEQ_LEN_MORE_CHUNKING=608 \
    TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$TREE" --size 640 --repeat 1 \
    --label rollback_640 --out "$OUT/rollback_640.json" > "$OUT/rollback_640.log" 2>&1
echo "EXIT = $?"
echo "--- reverted arm (assembled tree, both switches at pre-cutover values) ---"
grep -hE "cold|warm" "$OUT/rollback_640.log" | tail -2
echo "--- for comparison ---"
echo "  whpre 640 (pre-cutover tree)  : 07d7a43010403b18  plddt=0.867166"
echo "  whcut 640 (assembled, default): 41c6edf98868d66b  plddt=0.863318"
echo "ROLLBACK PROOF DONE $(date -u +%FT%TZ)"
