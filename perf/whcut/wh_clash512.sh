#!/bin/bash
# The one comparison the card race destroyed and nothing else replaces: whpre at 512 aa.
#
# The sweep scored whcut 512 as FAIL (6 heavy-atom clashes, 0.15 % of atoms, worst 1.701 A)
# but its whpre counterpart lost the race, so there is nothing to compare it against. That
# matters more than the 640 pair: 512 is BELOW K3's 448-960 band, so if whcut is worse there
# too, the cause is not K3 and the whole §6.3 reading changes.
#
# whcut 640 is not re-run -- §6.3 already measured it properly (5 marginal contacts, 0.097 %),
# and the sweep's whpre 640 reproduced §6.3's whpre 640 exactly (4 contacts, worst 1.308 A),
# which also confirms the fold is deterministic across separate invocations.
#
# Chained on the previous job's PID, never on card occupancy (§100).
set -u
TREE=/home/cust-team/mthuening/whbase/wt-whcut
PRE=/home/cust-team/mthuening/whbase/pxmain
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python3
OUT=$TREE/perf/whcut/out/wh/clash_sweep
PREV_PID=${PREV_PID:-1513932}
cd "$TREE" || exit 1
while kill -0 "$PREV_PID" 2>/dev/null; do sleep 30; done
sleep 20
echo "CLASH 512 CONTROL START $(date -u +%FT%TZ)"
tmp="$OUT/tmp_whpre_512b"; rm -rf "$tmp"; mkdir -p "$tmp"
stamp="$OUT/.stamp_whpre_512b"; touch "$stamp"; sleep 1
env TT_VISIBLE_DEVICES=28 TT_METAL_LOGGER_LEVEL=FATAL TMPDIR="$tmp" \
    TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$PRE" --size 512 --repeat 1 \
    --label cs_whpre_512b --out "$OUT/whpre_512b.json" > "$OUT/whpre_512b.log" 2>&1
echo "EXIT whpre 512 = $?"
cif=$(find "$tmp" -name "*.cif" -newer "$stamp" 2>/dev/null | head -1)
if [ -n "$cif" ]; then
  "$PY" perf/wh-correctness/check_structure.py "$cif" \
    --json "$OUT/whpre_512b_score.json" > "$OUT/whpre_512b_score.txt" 2>&1
  echo "=== whpre 512 ==="; head -3 "$OUT/whpre_512b_score.txt"
  echo "=== whcut 512 (for comparison) ==="; head -3 "$OUT/whcut_512_score.txt"
else
  echo "NO CIF FOUND"
fi
echo "CLASH 512 CONTROL DONE $(date -u +%FT%TZ)"
