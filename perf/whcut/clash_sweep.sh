#!/bin/bash
# Does the assembled tree systematically produce more marginal contacts, or was §6.3's
# "5 against 4" one target's response?
#
# §6.3 folded 640 aa on both trees and scored 5 marginal contacts on the assembled tree
# against 4 on the pre-cutover one, with the WORST contact less severe (1.431 A against
# 1.308 A) and backbone geometry unchanged. §87 established that the fold is DETERMINISTIC --
# cold and warm produced byte-identical CIFs in both arms -- so repeating a fold cannot
# separate signal from noise. Only more INPUTS can.
#
# So: three sizes on both trees, scored with the sweep's own checker. 512 is below K3's band
# and must be identical (its k pick does not move); 640 and 768 are inside it. If the
# assembled tree is worse at every size in the band that is a finding; if it scatters, the
# mean has not moved.
#
# Runs after the parity gate releases card 28. Waits on the card, not on a process pattern.
set -u
TREE=/home/cust-team/mthuening/whbase/wt-whcut
PRE=/home/cust-team/mthuening/whbase/pxmain
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python3
OUT=$TREE/perf/whcut/out/wh/clash_sweep
CARD=${CARD:-28}
mkdir -p "$OUT"
cd "$TREE" || exit 1
while [ "$(sudo -n lsof -t /dev/tenstorrent/4 2>/dev/null | wc -w)" != "0" ]; do sleep 60; done
echo "CLASH SWEEP START $(date -u +%FT%TZ) card $CARD"

arm() {  # label tree size
  local tmp="$OUT/tmp_$1_$3"; rm -rf "$tmp"; mkdir -p "$tmp"
  local stamp="$OUT/.stamp_$1_$3"; touch "$stamp"; sleep 1
  env TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL TMPDIR="$tmp" \
      TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
    "$PY" perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$2" --size "$3" --repeat 1 \
      --label "cs_$1_$3" --out "$OUT/$1_$3.json" > "$OUT/$1_$3.log" 2>&1
  local rc=$?
  local cif
  cif=$(find "$tmp" -name "*.cif" -newer "$stamp" 2>/dev/null | head -1)
  if [ -n "$cif" ]; then
    "$PY" perf/wh-correctness/check_structure.py "$cif" \
      --json "$OUT/${1}_${3}_score.json" > "$OUT/${1}_${3}_score.txt" 2>&1
    echo "$1 $3 rc=$rc  $(grep -hE 'contacts|clash' "$OUT/${1}_${3}_score.txt" | head -1)" \
         "$(grep -hE 'cif=' "$OUT/$1_$3.log" | tail -1 | sed 's/.*cif=//')"
  else
    echo "$1 $3 rc=$rc NO CIF FOUND"
  fi
}

for S in 512 640 768; do
  arm whcut "$TREE" "$S"
  arm whpre "$PRE"  "$S"
done
echo "CLASH SWEEP DONE $(date -u +%FT%TZ)"
echo "=== summary ==="
for S in 512 640 768; do
  for A in whcut whpre; do
    printf "%-6s %4s  %s\n" "$A" "$S" \
      "$(grep -hE 'contacts|clash' "$OUT/${A}_${S}_score.txt" 2>/dev/null | head -1 || echo 'clean')"
  done
done
