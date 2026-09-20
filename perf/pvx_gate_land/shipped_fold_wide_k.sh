#!/bin/bash
# The SHIPPED measurement for TT_BIO_SDPA_WIDE_K, to be run ON origin/main AFTER the merge.
#
# WHY IT REFUSES TO RUN ON THE WRONG ENGINE. The number that enters SHIPPED: is the one a user
# gets (verify-the-deployed-artifact-not-your-own-change). The folds below import tt_bio from $WT
# through PYTHONPATH, so what actually gets scored is the WORKING TREE, not the commit HEAD points
# at. The first version of this guard compared `git rev-parse HEAD` to origin/main, which is the
# wrong object in both directions: it PASSES with uncommitted edits to tt_bio/ sitting in the tree
# (scoring code no user has), and it REFUSES a worktree whose engine is byte-identical to main
# because its HEAD is a branch. The guard now diffs the working tree against origin/main over the
# engine paths, which is the claim "this is what a user gets" stated about the files that will be
# imported. The default is checked separately, from the package that import resolves.
#
# WHY 352 AND 1088 AND NOT 512. The flag changes the k-chunk pick only at the twenty padded lengths
# whose shipped chunk does not divide them, and 512 is not one: _dividing_sdpa_chunk_size
# short-circuits on `padded % cap == 0`, SDPA_CHUNK_MAX is 256, 512 % 256 == 0. At 512 aa this
# lever is 0.0000 s and always will be. Of the committed fixtures, 352, 385, 400, 416, 1088 and 1300
# land on an affected padded length; 352 is the cheapest and 1088 is the one that is also a
# size-ladder rung. The script recomputes that set from tt_bio rather than trusting this comment,
# and refuses a rung that does not fire -- a fold at a non-firing length would read 0.0000 s and
# look like a refuted lever instead of a mis-chosen cell.
#
# THE PREDICTION, written before the folds (perf-method-floor-screen-predict-then-build).
# The op is 1.27x-4.39x where it fires and Protenix-v2's trunk stage read 1.1285x at padded 704
# (120.0 -> 106.3 s, 1208 calls served). The trunk is part of a fold, so the fold-level figure must
# be SMALLER than the stage figure. Predicted: a few per cent at 352 aa, where the shipped k is 64
# and the wide pick is a single 352 chunk, and less at 1088, where the shipped pick already has 256
# available and the wide ladder is [1088, 544, 256]. A reading above 1.1285x at the fold level would
# mean the cell is measuring something other than this lever. Anything inside the A/A floor is
# 0.0000 s and gets reported as such.
#
# The A/A floor, the interleave and the discarded cold fold come from perf/xmsoftmax/fold_ab_flip.py,
# which is the committed protocol; this script only pins the flag, the cell and the environment.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=${CARD:-1}
RUNGS=${RUNGS:-352,1088}
REPS=${REPS:-2}
OUT="$WT/perf/pvx_gate_land/shipped_fold_wide_k.json"
LOG="$WT/perf/pvx_gate_land/shipped_fold_wide_k.log"
CONT="$WT/perf/pvx_gate_land/shipped_fold_contention.jsonl"

cd "$WT" || exit 1

git fetch -q origin || exit 1
if ! git diff --quiet origin/main -- tt_bio scripts; then
  echo "REFUSING: the engine in $WT is not origin/main's. Differences that would be scored:"
  git diff --stat origin/main -- tt_bio scripts
  echo "The shipped number is measured on what a user gets. Land the change first, then run this."
  exit 1
fi
echo "engine check: tt_bio and scripts in $WT are byte-identical to origin/main $(git rev-parse --short origin/main)"

PYTHONPATH="$WT" $PY - "$RUNGS" <<'PYEOF' || exit 1
import sys
sys.path.insert(0, ".")
import tt_bio
from tt_bio import tenstorrent as T
# Name the package that was actually scored. A wrong cwd resolves tt_bio to whatever is installed
# in the env instead of the checkout, and the refusal below then reports a default that belongs to
# a different tree (parity-gate-scores-installed-package-not-checkout).
print(f"scoring tt_bio from {tt_bio.__file__}")
if not T._SDPA_WIDE_K_DEFAULT or not T._sdpa_wide_k():
    print(f"REFUSING: default={T._SDPA_WIDE_K_DEFAULT} _sdpa_wide_k()={T._sdpa_wide_k()} in "
          f"{tt_bio.__file__}; the flag is not on by default in the package this would score, so "
          f"there is nothing shipped to measure")
    sys.exit(1)
pad = T.PAIRFORMER_PAD_MULTIPLE
import os
def ladder(n, wide):
    os.environ["TT_BIO_SDPA_WIDE_K"] = "1" if wide else "0"
    return T._tri_att_k_chunks(n, n)
affected = {n for n in range(T.SDPA_CHUNK_TILE, 1601, T.SDPA_CHUNK_TILE)
            if ladder(n, True) != ladder(n, False)}
bad = []
for r in [int(x) for x in sys.argv[1].split(",")]:
    padded = ((r + pad - 1) // pad) * pad
    fires = padded in affected
    print(f"rung {r} -> padded {padded}: {'FIRES' if fires else 'does not fire'}")
    if not fires:
        bad.append(r)
if bad:
    print(f"REFUSING: rungs {bad} do not reach an affected padded length, so they would read "
          f"0.0000 s for a reason that is not the lever")
    sys.exit(1)
print(f"wide-k default on, {len(affected)} affected lengths, every requested rung fires")
PYEOF

$PY "$WT/perf/pvx_gate_land/sample_contention.py" > "$CONT" 2>/dev/null &
SAMPLER=$!
trap 'kill '"$SAMPLER"' 2>/dev/null' EXIT

TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:pvx-gate-land \
TT_BIO_AICLK=1350 PYTHONPATH="$WT" \
  bash "$BL" pvx-gate-land -- "$PY" perf/xmsoftmax/fold_ab_flip.py \
    --models protenix-v2 --rungs "$RUNGS" --reps "$REPS" \
    --flag TT_BIO_SDPA_WIDE_K --off-value 0 \
    --workdir /tmp/widek_shipped --out "$OUT" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "fold_ab_flip rc=$rc"
# rc after a pipeline is the last command's status, so it is taken from PIPESTATUS and the artifact
# is asserted rather than inferred (rc-after-a-pipeline-is-the-last-commands-status).
[ -s "$OUT" ] || { echo "NO ARTIFACT at $OUT -- nothing measured"; exit 1; }
$PY - "$OUT" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for c in d["cells"]:
    if "off_median" not in c:
        print(f"{c['model']} {c['rung']}: FAILED {c.get('error','')[:70]}"); continue
    print(f"{c['model']} {c['rung']}aa  off {c['off_median']:.4f}s  on {c['on_median']:.4f}s  "
          f"A/A {c['aa_spread_pct']:+.3f}%  A/B {c['ab_median_pct']:+.3f}%  "
          f"{'INSIDE THE FLOOR -> 0.0000 s' if c['inside_aa'] else 'outside the floor'}")
PYEOF
echo "AICLK during the folds (every card, sampled every 30 s): $CONT"
exit $rc
