#!/bin/bash
# Stage 3: everything the Wormhole side still owes, on the ONE usable card.
#
# UMD 28 is the only spare that opens. 26 throws ARC, 27 and 30 hang on open at both 180 s
# and 300 s, 31 has been wedged since 08-16, and 29 I wedged myself by SIGTERMing a fold
# mid-device-operation (§63). No card is reset: this box is production, shared with a
# customer. So there is nothing to fan across and every step here is sequential.
#
#   1. the three ESMFold2 rows that did not run (two died with card 29, one never started)
#   2. §6.3, the 640 aa clash check, both arms on the same card one after the other
#   3. §6.2, the Wormhole parity gate, single worker
#
# Step 3 is expected to run long on one card -- the plan budgeted ~6 h across two or three
# workers. It is started anyway because every option open to this task benefits from as much
# of the gate being done as possible, and stopping it early yields a named subset rather than
# nothing.
set -u
TREE=/home/cust-team/mthuening/whbase/wt-whcut
PRE=/home/cust-team/mthuening/whbase/pxmain
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python3
OUT=$TREE/perf/whcut/out/wh
CARD=${CARD:-28}
mkdir -p "$OUT/clash640"
cd "$TREE" || exit 1

# Wait for the card itself to be free, not for a process pattern: a pattern also matches the
# status-check shells that carry it in their own command line.
while [ "$(sudo -n lsof -t /dev/tenstorrent/4 2>/dev/null | wc -w)" != "0" ]; do sleep 30; done
echo "STAGE3 START $(date -u +%FT%TZ) card $CARD head $(git rev-parse HEAD)"

run_esm() {  # model size
  echo "=== $1 $2 start $(date -u +%H:%M:%S)"
  env TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL \
      TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
    "$PY" -u perf/wh-esmfold2/fold_ab512.py --model "$1" --size "$2" --fast \
      --arms base,A --rounds 1 --out "$OUT/esm_$1_$2.json" > "$OUT/esm_$1_$2.log" 2>&1
  echo "EXIT $1 $2 = $?  $(grep -hE 'warm:' "$OUT/esm_$1_$2.log" | tail -2 | tr '\n' ' ')"
}
run_esm esmfold2-fast 1024
run_esm esmfold2-fast 512
run_esm esmfold2 512
echo "ESMFOLD2 ROWS DONE $(date -u +%FT%TZ)"

# §6.3. Sweep finding 0.9: Boltz-2 returns zero sub-2.0 A heavy-atom pairs at 128/256/512 and
# nine at 640 on this same tiled CDK2 fixture, and 640 is where K3 and lever C fire. Fold it on
# both trees and score both with the sweep's own checker. ACCEPT iff whcut is no worse.
# Sequential now, so the shared-TMPDIR hazard of §51 cannot arise at all -- each arm's fold is
# the only one running -- but the private TMPDIR is kept because it costs nothing.
clash_arm() {  # label tree
  local tmp="$OUT/clash640/tmp_$1"; rm -rf "$tmp"; mkdir -p "$tmp"
  local stamp="$OUT/clash640/.stamp_$1"; touch "$stamp"; sleep 1
  env TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL TMPDIR="$tmp" \
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
clash_arm whcut "$TREE"
clash_arm whpre "$PRE"
echo "CLASH 640 DONE $(date -u +%FT%TZ)"
grep -hiE "clash|gap|backbone" "$OUT/clash640"/*_score.txt 2>/dev/null | head -20

echo "WH PARITY START $(date -u +%FT%TZ)"
ESM_ROOT=/home/cust-team/mthuening/esm \
RELEASE_GATE_MSA_DIR=/home/cust-team/mthuening/abag_xm/msa_cache \
PYTHONPATH="$TREE" TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/full_parity_gate.py \
    --workers UF-EV-A13-GWH02:$CARD \
    --fold-timeout 4800 --fresh --workdir "$TREE/perf/whcut/out/parity-wh" \
    --leg boltz2-hsa-nomsa --leg boltz2-trpcage-nomsa --leg boltz2-prot-nomsa \
    --leg protenix-prot-msa --leg protenix-ubq-msa --leg protenix-hsa-msa \
    --leg esmfold2-trpcage --leg opendde-trpcage-nomsa \
    --leg esmc-300m --leg esmc-600m --leg saprot-35m --leg saprot-650m \
    --leg boltzgen --leg rfd3-featurizer
echo "STAGE3 EXIT $? $(date -u +%FT%TZ)"
