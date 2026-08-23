#!/usr/bin/env bash
# Refresh of the RF3 512 aa page cell on the host that produced it: qb2, ttnn 0.68.0, 11x10 grid,
# card 2 (where the published 82.547 s was taken). Page protocol unchanged: page512_tt.py, --arm a0
# (ARMS["a0"] == {}, so it overrides nothing and measures the shipped default), --repeat 2, two
# independent processes, cold fold discarded, benchlock.
set -u
WT=/home/ttuser/.coworker/wt/rf3-perf-page-row-refresh
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PP="$WT:/home/ttuser/rf3_perf_deps"
OUT=$WT/perf/rf3/page512
CARD=2
HOST=qb2c2
LEASE=$HOME/.coworker/state/leases/tt-quietbox2-card$CARD.json
cd "$WT" || exit 1

# The cell has to price the tree that ships. main merged twice while pass 4's first process was
# running, so this is asserted at launch rather than assumed from whenever the branch was cut:
# tt_bio must be byte-identical to origin/main, and the SHA it matched goes in the provenance file
# next to the results so the `ref` can name it instead of naming the branch.
git fetch origin -q || { echo "cannot fetch origin, refusing to measure"; exit 70; }
MAIN=$(git rev-parse origin/main)
if [ -n "$(git diff --name-only "$MAIN" -- tt_bio/)" ]; then
  echo "REFUSING: tt_bio differs from origin/main ($MAIN):"
  git diff --stat "$MAIN" -- tt_bio/
  exit 71
fi
mkdir -p "$OUT"
cat > "$OUT/provenance_$HOST.json" <<PROV
{"origin_main": "$MAIN",
 "head": "$(git rev-parse HEAD)",
 "tt_bio_identical_to_main": true,
 "non_tt_bio_diff_vs_main": [$(git diff --name-only "$MAIN" | grep -v '^perf/' | sed 's/.*/"&"/' | paste -sd, -)],
 "host": "$(hostname)", "card": $CARD, "checked_at": "$(date -u +%FT%TZ)"}
PROV
echo "tt_bio is byte-identical to origin/main $MAIN"

# 3600 was not enough: pass 4's p2 queued behind a chain with two `timeout 1800` legs left,
# so its wait would have expired before the lock freed and benchlock.sh exits 75 there
# rather than measuring. Overridable, because a leg that waits all night is also wrong.
export BENCHLOCK_LOAD_WAIT_S=60 BENCHLOCK_WAIT_S=${BENCHLOCK_WAIT_S:-9000}
# benchlock.sh's default FOREIGN_RE predates half the model set; name the rest explicitly.
export BENCHLOCK_FOREIGN_RE="fold_ab512|tt_baseline|protenix|boltz|opendde|esmfold|openfold|openbind|nesso|rfd3|rf3|pxtrace|host_cost_probe|release_gate|lever_census|pxdesign|af2ig"

card_free() {
  [ -f "$LEASE" ] || return 0
  "$PY" -c "
import json,os,sys
try: d=json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
if d.get('released') is not None: sys.exit(0)
pid=d.get('pid')
sys.exit(0 if not pid or not os.path.exists('/proc/%s' % pid) else 1)
" "$LEASE"
}
foreign() {
  ps -eo pid,args | grep -E "$BENCHLOCK_FOREIGN_RE" | grep python \
    | grep -vE "claude|cursor|worker\.sh|benchlock|grep|page512_tt" | grep -v "^ *$$ "
}
nforeign() { foreign | wc -l; }

for i in $(seq 1 60); do
  card_free && break
  echo "waiting for card $CARD lease ($i): $(cat "$LEASE" 2>/dev/null)"
  sleep 20
done
card_free || { echo "card $CARD still held after 20 min, giving up"; exit 75; }
echo "card $CARD lease free at $(date -u +%FT%TZ)"

# Wait for quiet OUTSIDE the flock (benchlock takes its lock first and has no fairness, so waiting
# from inside it starves every other timed run on the box). Bounded: if the box will not go quiet we
# measure anyway and record what was running, and the reading is then an upper bound, not a lie.
QUIET=0
for i in $(seq 1 26); do
  L=$(cut -d' ' -f1 /proc/loadavg); NF=$(nforeign)
  if [ "$NF" -eq 0 ] && awk -v a="$L" 'BEGIN{exit !(a+0<=2.0)}'; then QUIET=1; break; fi
  echo "loud ($i): load $L, $NF foreign at $(date -u +%FT%TZ)"
  sleep 25
done
echo "QUIET=$QUIET at $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"

for P in ${RF3_LEGS:-p1 p2}; do
  if [ -f "$OUT/postflip_${HOST}_${P}.json" ]; then
    echo "=== $P already has a result, skipping (delete it to re-measure) ==="
    continue
  fi
  NF=$(nforeign)
  echo "=== $P start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) foreign=$NF ==="
  foreign | sed 's/^/    cotenant: /'
  # publish_cell.py requires this count and refuses to infer it: the per-fold loadavg is sampled
  # at fold END and read 1.96 for a leg that started with six foreign folds running.
  echo "{\"leg\": \"$P\", \"foreign_at_start\": $NF, \"loadavg\": \"$(cut -d' ' -f1-3 /proc/loadavg)\"}" \
    > "$OUT/cotenancy_${HOST}_${P}.json"
  /home/ttuser/.coworker/scripts/benchlock.sh rf3-perf-page-row-refresh -- \
    env PYTHONPATH="$PP" \
        TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
        TT_BIO_LEASE_HOLDER=worker:rf3-perf-page-row-refresh \
        TT_BIO_SDPA_RAGGED_CENSUS="$OUT/census_postflip_${HOST}_${P}" \
      "$PY" perf/rf3/page512_tt.py --repeat 2 --arm a0 \
        --label "postflip_default_${HOST}_${P}" \
        --out "$OUT/postflip_${HOST}_${P}.json" \
        > "$OUT/postflip_${HOST}_${P}.log" 2>&1
  echo "=== $P exit $? $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) ==="
  tail -3 "$OUT/postflip_${HOST}_${P}.log"
done
echo "ALLDONE $(date -u +%FT%TZ)"
