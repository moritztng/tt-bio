#!/usr/bin/env bash
# qb1 D3 datum for the RF3 512 aa page cell: the shipped default (fused triangle attention) at the
# page protocol, then the pre-flip materialised control on the same card.
#
# This is NOT the page cell. The cell was measured on qb2 at ttnn 0.68.0 on an 11x10 grid; qb1 is a
# different board subsystem, a 13x10 grid and ttnn 0.67.4. See ~/.coworker/state/rf3-perf-page-cell.md
# D1/D3. A qb1 number must never be published as a refresh of that cell.
#
# Three guards, all learned on 2026-08-23 between 15:52 and 15:58 UTC:
#
#  1. CARD. A card grant is not exclusive. The dispatcher handed card 2 to this task and to
#     worker:protenix-v2-trunk-dispatch-trace at the same minute, and tt_bio CardSetLease refused
#     the open after waiting 120 s. Wait for the lease to be genuinely free first. released=null
#     with a dead holder pid is a stale lease, not a live one.
#  2. BENCHLOCK FOREIGN_RE. The benchlock.sh default is
#     fold_ab512|tt_baseline|protenix|boltz|opendde|esmfold|openfold, which matches none of the
#     models added since: openbind, nesso, rfd3, rf3 itself, nor the release_gate size-ladder driver
#     or a pxtrace host_cost_probe. It green-lit this run while an openbind 640 fold was live on
#     card 1. Name them explicitly.
#  3. WAIT OUTSIDE THE LOCK. benchlock takes its flock and only then waits for quiet, and it has no
#     fairness. Waiting out a 90 min size-ladder from inside the lock starves every other timed
#     measurement on the host, so do that wait before acquiring.
set -u
WT=/home/ttuser/.coworker/wt/rf3-perf-page-row-refresh
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PP="$WT:/home/ttuser/rf3_perf_deps"
OUT=$WT/perf/rf3/page512
CARD="${CARD:-2}"
HOST="${HOSTTAG:-qb1c$CARD}"
LEASE=$HOME/.coworker/state/leases/tt-quietbox-card$CARD.json
cd "$WT" || exit 1

export BENCHLOCK_LOAD_WAIT_S=600 BENCHLOCK_WAIT_S=7200
export BENCHLOCK_FOREIGN_RE="fold_ab512|tt_baseline|protenix|boltz|opendde|esmfold|openfold|openbind|nesso|rfd3|rf3|pxtrace|host_cost_probe|release_gate|lever_census"

card_free() {
  [ -f "$LEASE" ] || return 0
  "$PY" -c "
import json,os,sys
try: d=json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
if d.get(\"released\") is not None: sys.exit(0)
pid=d.get(\"pid\")
sys.exit(0 if not pid or not os.path.exists(\"/proc/%s\" % pid) else 1)
" "$LEASE"
}

box_quiet() {
  local l nf
  l=$(cut -d" " -f1 /proc/loadavg)
  nf=$(ps -eo pid,args | grep -E "$BENCHLOCK_FOREIGN_RE" | grep python \
       | grep -vE "claude|cursor|worker\.sh|benchlock|grep" | wc -l)
  [ "$nf" -eq 0 ] && awk -v a="$l" "BEGIN{exit !(a+0<=2.0)}"
}

for i in $(seq 1 240); do
  card_free && break
  echo "waiting for card $CARD ($i): $(cat "$LEASE" 2>/dev/null)"
  sleep 30
done
card_free || { echo "card $CARD still held after 2 h, giving up"; exit 75; }

for i in $(seq 1 300); do
  box_quiet && break
  echo "box loud ($i): load $(cut -d" " -f1 /proc/loadavg) at $(date -u +%FT%TZ)"
  sleep 30
done
# Refuse rather than measure dirty. A suspect D3 datum is worse than no datum: its whole job is
# to price this column on the board the page names, and a co-tenanted reading cannot do that.
box_quiet || { echo "box never went quiet in 150 min; refusing to measure. Not a slow number, a wrong one."; exit 75; }

run_leg() {  # run_leg <tag> <outstem> [extra env assignments...]
  local tag=$1 stem=$2; shift 2
  for P in p1 p2; do
    echo "=== $tag $P start $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg) ==="
    /home/ttuser/.coworker/scripts/benchlock.sh rf3-perf-page-row-refresh -- \
      env PYTHONPATH="$PP" \
          TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
          TT_BIO_LEASE_HOLDER=worker:rf3-perf-page-row-refresh \
          "$@" \
        "$PY" perf/rf3/page512_tt.py --repeat 2 --arm a0 \
          --label "${stem}_${HOST}_${P}" \
          --out "$OUT/${stem}_${HOST}_${P}.json" \
          > "$OUT/${stem}_${HOST}_${P}.log" 2>&1
    echo "=== $tag $P exit $? $(date -u +%FT%TZ) ==="
  done
}

# Shipped default. a0 overrides nothing (ARMS["a0"] == {}), so this measures whatever the tree
# ships, which post-flip is the fused route. The "arm": {} in the output JSON is the proof.
run_leg fused postflip TT_BIO_SDPA_RAGGED_CENSUS="$OUT/census_postflip_${HOST}"

# Pre-flip materialised control, same card, same fixture: the route the published 82.547 s cell was
# on. BOLTZ2_FP32_SOFTMAX=1 is the global _attend_heads reads (tt_bio/tenstorrent.py:169). This pair
# prices the flip on 13x10 without needing the qb2 cell as its denominator, which is the only way a
# qb1 reading says anything about the flip at all.
run_leg matz preflip_mat BOLTZ2_FP32_SOFTMAX=1

echo "ALLDONE $(date -u +%FT%TZ)"
