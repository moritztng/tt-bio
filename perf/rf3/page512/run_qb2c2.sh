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
# A plain diff against main cannot tell "we edited this file" from "main moved past us since the
# branch last merged", and the second is not a reason to distrust a fold. Record both sides off the
# merge base: `ours` is what this branch actually changed, which is what the publish gate reads,
# `theirs_since_base` is main moving on. The first refused a publish that nothing was wrong with.
BASE=$(git merge-base HEAD "$MAIN")
mkdir -p "$OUT"
cat > "$OUT/provenance_$HOST.json" <<PROV
{"origin_main": "$MAIN",
 "head": "$(git rev-parse HEAD)",
 "merge_base": "$BASE",
 "tt_bio_identical_to_main": true,
 "ours_vs_merge_base": [$(git diff --name-only "$BASE" HEAD | grep -v '^perf/' | sed 's/.*/"&"/' | paste -sd, -)],
 "theirs_since_merge_base": [$(git diff --name-only "$BASE" "$MAIN" | sed 's/.*/"&"/' | paste -sd, -)],
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
  RES="$OUT/postflip_${HOST}_${P}.json"
  if [ -f "$RES" ]; then
    if [ "$(cat "$RES.main" 2>/dev/null)" = "$MAIN" ]; then
      echo "=== $P already has a result measured at $MAIN, skipping ==="
      continue
    fi
    # Unstamped or stamped against another tree. Quarantine rather than skip and rather than
    # overwrite: a stale result silently pooled with a fresh one is the failure this guards.
    mkdir -p "$OUT/stale"
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    mv -f "$RES" "$OUT/stale/$(basename "$RES").$STAMP"
    # The census dir goes with it: the fresh run writes ragged_sites_<pid>.json into the same
    # directory, so a leftover file there would be globbed into the publish as an extra process.
    [ -d "$OUT/census_postflip_${HOST}_${P}" ] &&
      mv -f "$OUT/census_postflip_${HOST}_${P}" "$OUT/stale/census_${P}.$STAMP"
    echo "=== $P had a result from another tree ($(cat "$RES.main" 2>/dev/null || echo unstamped)),"
    echo "    quarantined under $OUT/stale, re-measuring ==="
  fi
  echo "=== $P queued $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) foreign=$(nforeign) ==="
  foreign | sed 's/^/    queued behind: /'
  # The co-tenancy snapshot publish_cell.py reads is taken by leg_inner.sh, INSIDE the lock, where
  # the timed fold actually starts. Taken here it describes a leg that is still waiting for the
  # flock: this pass queued behind a 45-minute encoder A/B, and a count from here would have said
  # "three foreign folds" about a leg that then ran on a quiet box.
  /home/ttuser/.coworker/scripts/benchlock.sh rf3-perf-page-row-refresh -- \
    bash "$OUT/leg_inner.sh" "$OUT" "$HOST" "$P" "$CARD" "$PP" "$PY" \
        > "$OUT/postflip_${HOST}_${P}.log" 2>&1
  RC=$?
  echo "=== $P exit $RC $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) ==="
  [ "$RC" = 0 ] && [ -f "$RES" ] && printf '%s' "$MAIN" > "$RES.main"
  tail -3 "$OUT/postflip_${HOST}_${P}.log"
done
echo "ALLDONE $(date -u +%FT%TZ)"
