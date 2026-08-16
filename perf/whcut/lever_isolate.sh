#!/bin/bash
# Which lever causes the 640 aa contact change -- K3 or lever C?
#
# §117 showed both fire at 640 (K3 moves the k pick 256->160; lever C flips 640 from the
# chunked path to the unchunked one) while only lever C fires at 768. §116 showed turning
# BOTH off recovers whpre's bytes exactly. So the two switches account for the whole
# difference; this splits it between them.
#
# It matters for risk, not curiosity: lever C returns early on any grid >= 110 cores and so
# is inert on Blackhole by construction, whereas K3 changes output on BOTH architectures. If
# the contact reshuffle is lever C's, it cannot occur on Blackhole at all.
#
#   both on   (whcut default)  41c6edf98868d66b  plddt 0.863318  5 contacts
#   both off  (§116)           07d7a43010403b18  plddt 0.867166  4 contacts  == whpre
set -u
TREE=/home/cust-team/mthuening/whbase/wt-whcut
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python3
OUT=$TREE/perf/whcut/out/wh/lever_isolate
CARD=${CARD:-28}
mkdir -p "$OUT"
cd "$TREE" || exit 1
echo "LEVER ISOLATE START $(date -u +%FT%TZ)"

arm() {  # label  K3value  leverCvalue
  local tmp="$OUT/tmp_$1"; rm -rf "$tmp"; mkdir -p "$tmp"
  local stamp="$OUT/.stamp_$1"; touch "$stamp"; sleep 1
  env TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL TMPDIR="$tmp" \
      TT_BIO_SDPA_DIV_K=$2 TT_BIO_SEQ_LEN_MORE_CHUNKING=$3 \
      TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
    "$PY" perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$TREE" --size 640 --repeat 1 \
      --label "$1" --out "$OUT/$1.json" > "$OUT/$1.log" 2>&1
  echo "EXIT $1 = $?  $(grep -hE 'warm 0' "$OUT/$1.log" | tail -1)"
  local cif
  cif=$(find "$tmp" -name "*.cif" -newer "$stamp" 2>/dev/null | head -1)
  [ -n "$cif" ] && "$PY" perf/wh-correctness/check_structure.py "$cif" \
      --json "$OUT/$1_score.json" > "$OUT/$1_score.txt" 2>&1
  grep -hE "FAIL|WARN|PASS" "$OUT/$1_score.txt" 2>/dev/null | head -2 | sed 's/^/     /'
}

# K3 off, lever C ON (1088 = the assembled value)
arm k3off_leverC_on 0 1088
# K3 ON, lever C off (608 = the pre-cutover value)
arm k3on_leverC_off 1 608
echo "LEVER ISOLATE DONE $(date -u +%FT%TZ)"
