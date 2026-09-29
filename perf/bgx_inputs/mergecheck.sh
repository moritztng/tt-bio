#!/bin/bash
# Verify this branch against main's current head without merging it, in a throwaway worktree.
#
# It carries the card pin. That is the point of the file: the first version of this check ran
# host-only suites, someone (me) later added `tests/test_bindcraft2.py` to it, and one test in
# there opens a device -- so an unpinned merge check took /dev/tenstorrent/0 and 1 while two
# other rows were on them (state/notice-bgx-inputs-unpinned-open-on-cards-0-and-1.md). A script
# that can open a card carries the lease, or it does not run the suite that opens one.
set -uo pipefail
CARD=${CARD:-3}
TREE=${TREE:-/tmp/bgx_mergecheck}
cd "$(dirname "$0")/../.."
REPO=$PWD
git fetch -q origin main
MAIN=$(git rev-parse origin/main)
BRANCH=$(git rev-parse wk/bgx-inputs)
echo "origin/main      $MAIN"
echo "wk/bgx-inputs    $BRANCH"
rm -rf "$TREE"
git worktree add -q --detach "$TREE" "$MAIN" || exit 1
trap 'cd "$REPO"; git worktree remove --force "$TREE" >/dev/null 2>&1' EXIT
cd "$TREE"
git -c user.name=moritztng -c user.email=moritz.thuening@gmail.com merge --no-edit -q "$BRANCH" \
    && echo "MERGE CLEAN" \
    || { echo "MERGE CONFLICT"; git diff --name-only --diff-filter=U; exit 1; }

export PYTHONPATH=$PWD:/home/ttuser/bcx_e2e/bc2
P=/home/ttuser/bcx_e2e_venv/bin/python3
echo "=== host-only suites at the merged tree (no card touched)"
timeout 600 $P -m pytest tests/test_bcinputs.py -q 2>&1 | tail -2
timeout 900 $P -m pytest tests/test_citations.py tests/test_perf_citations.py \
    tests/test_capabilities_doc.py tests/test_shipped_models_are_documented.py \
    tests/test_tuning_flag_docs.py -q 2>&1 | tail -2
echo "=== device suite at the merged tree, pinned to card $CARD"
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bgx-inputs \
    timeout 1800 $P -m pytest tests/test_bindcraft2.py -q 2>&1 | tail -2
