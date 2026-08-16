#!/bin/bash
# Stage 2 on Wormhole, sequenced behind the §5.1 fold chain on the same two cards:
#   1. the four ESMFold2 rows, on the harness their reference numbers came from
#   2. §6.3, the 640 aa clash check against the pre-cutover tree
#   3. §6.2, the parity gate
# Sequenced rather than stacked: three jobs on qb1s four cards cost the Blackhole perf gate
# two legs this morning.
#
# WHY THE ESMFOLD2 RE-RUN. §5.1 labels both rows "esmfold2-fast --fast" and asks for 83.843 s
# at 512. esmfold2-fast is a different CHECKPOINT (24 trunk blocks, no MSA encoder), not the
# --fast flag; 83.843 s is the esmfold2 checkpoints number and esmfold2-fast is 49.576 s
# there; and every reference came from perf/wh-esmfold2/fold_ab512.py, not xmodel_ab.py.
#
#   model          size  must reproduce (was)
#   esmfold2        512   83.843  (93.512)
#   esmfold2       1024  367.032 (395.224)
#   esmfold2-fast   512   49.576  (54.630)
#   esmfold2-fast  1024  203.679 (220.374)
#
# fold_ab512.py arms are overrides against the SHIPPED default, so `base` on this tree is
# what a user now gets and it should land on the recorded `A` column.
set -u
TREE=/home/cust-team/mthuening/whbase/wt-whcut
PRE=/home/cust-team/mthuening/whbase/pxmain
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python3
OUT=$TREE/perf/whcut/out/wh
CHAIN_PID=${CHAIN_PID:-773336}
mkdir -p "$OUT/clash640"
cd "$TREE" || exit 1
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done
echo "STAGE2 START $(date -u +%FT%TZ) head $(git rev-parse HEAD)"

run_esm() {  # model size card
  echo "=== $1 $2 card $3 start $(date -u +%H:%M:%S)"
  env TT_VISIBLE_DEVICES=$3 TT_METAL_LOGGER_LEVEL=FATAL \
      TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
    "$PY" -u perf/wh-esmfold2/fold_ab512.py --model "$1" --size "$2" --fast \
      --arms base --rounds 1 --out "$OUT/esm_$1_$2.json" > "$OUT/esm_$1_$2.log" 2>&1
  echo "EXIT $1 $2 = $?"
}
( run_esm esmfold2 1024 28; run_esm esmfold2 512 28 ) > "$OUT/stage2_c28.log" 2>&1 &
P1=$!
( run_esm esmfold2-fast 1024 29; run_esm esmfold2-fast 512 29 ) > "$OUT/stage2_c29.log" 2>&1 &
P2=$!
wait $P1 $P2
echo "ESMFOLD2 ROWS DONE $(date -u +%FT%TZ)"

# §6.3. Sweep finding 0.9: Boltz-2 returns zero sub-2.0 A heavy-atom pairs at 128/256/512 and
# nine at 640 (0.18 % of atoms, worst 1.521 A) on this same tiled CDK2 fixture. 640 is exactly
# where K3 and lever C fire, so fold it on both trees and score both with the sweeps own
# checker. ACCEPT iff whcut is no worse than whpre.
#
# Each arm gets a PRIVATE TMPDIR. tt_baseline builds struct_dir with tempfile.mkdtemp under
# TMPDIR (scripts/gpu_vs_tt/tt_baseline.py:158) and both arms use the same "ttbase-boltz2-"
# prefix, so with a shared /tmp each arms `find` could pick up the OTHER trees CIF and score
# a structure against itself -- a comparison that cannot fail and therefore means nothing.
clash_arm() {  # label tree card
  local tmp="$OUT/clash640/tmp_$1"; rm -rf "$tmp"; mkdir -p "$tmp"
  local stamp="$OUT/clash640/.stamp_$1"; touch "$stamp"; sleep 1
  env TT_VISIBLE_DEVICES=$3 TT_METAL_LOGGER_LEVEL=FATAL TMPDIR="$tmp" \
      TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
    "$PY" perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$2" --size 640 --repeat 1 \
      --label "clash_$1" --out "$OUT/clash640/$1.json" > "$OUT/clash640/$1.log" 2>&1
  echo "EXIT clash $1 = $?"
  find "$tmp" -name "*.cif" -newer "$stamp" 2>/dev/null | head -3 | while read -r c; do
    cp "$c" "$OUT/clash640/$1_$(basename "$c")"
    "$PY" perf/wh-correctness/check_structure.py "$c" \
      --json "$OUT/clash640/$1_score.json" >> "$OUT/clash640/$1_score.txt" 2>&1
  done
}
clash_arm whcut "$TREE" 28 &
clash_arm whpre "$PRE"  29 &
wait
echo "CLASH 640 DONE $(date -u +%FT%TZ)"

echo "WH PARITY START $(date -u +%FT%TZ)"
ESM_ROOT=/home/cust-team/mthuening/esm \
RELEASE_GATE_MSA_DIR=/home/cust-team/mthuening/abag_xm/msa_cache \
PYTHONPATH="$TREE" TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/full_parity_gate.py \
    --workers UF-EV-A13-GWH02:28,UF-EV-A13-GWH02:29 \
    --fold-timeout 4800 --fresh --workdir "$TREE/perf/whcut/out/parity-wh" \
    --leg esmc-300m --leg esmc-600m --leg saprot-35m --leg saprot-650m \
    --leg esmfold2-trpcage --leg boltz2-trpcage-nomsa --leg boltz2-prot-nomsa \
    --leg boltz2-prot-msa --leg boltz2-ubiquitin-msa --leg boltz2-hsa-nomsa \
    --leg protenix-prot-msa --leg protenix-ubq-msa --leg protenix-hsa-msa \
    --leg opendde-trpcage-nomsa --leg boltzgen --leg rfd3-featurizer
echo "STAGE2 EXIT $? $(date -u +%FT%TZ)"
