#!/bin/bash
# Check a whglx worktree out to this branch's head, but not underneath a running measurement.
#
# BoltzGen spawns a fresh subprocess for every one of its six steps, so a checkout under a
# live job is not inert: the next step imports whatever is on disk then. A job that started
# on one engine and finished on another is unattributable, and nothing in its output says so.
#
# I did this by hand once and got away with it only because the diff happened to be perf/
# files -- checked AFTER the checkout, which is the wrong order. So the rule is mechanical:
# if a device job is running out of the tree AND the incoming diff touches engine paths,
# refuse. If the diff is plans and results only, a running job cannot see it and the sync is
# safe. --force overrides, for the case where the job is known to be finished with imports.
#
#   bash perf/mgxaccuracy/sync_tree.sh REMOTE_TREE [--force]
set -u
TREE=${1:?remote tree path on whglx, e.g. \$HOME/wt-mgx-catcher}
FORCE=${2:-}
ENGINE='^(tt_bio|scripts|tests)/'

read -r head busy <<<"$(ssh -o BatchMode=yes whglx "cd $TREE \
    && git fetch -q origin wk/mgx-design-accuracy \
    && printf '%s %s' \"\$(git rev-parse HEAD)\" \
       \"\$(pgrep -cf \"\$(cd $TREE && pwd)/perf\" || true)\"")"
[ -n "${head:-}" ] || { echo "REFUSED: could not read $TREE on whglx"; exit 1; }

incoming=$(ssh -o BatchMode=yes whglx "cd $TREE && git diff --name-only HEAD origin/wk/mgx-design-accuracy")
if [ -z "$incoming" ]; then echo "already at the branch head ($head), nothing to do"; exit 0; fi

touches_engine=$(printf '%s\n' "$incoming" | grep -cE "$ENGINE" || true)
echo "tree $TREE at $head"
echo "  incoming files: $(printf '%s\n' "$incoming" | wc -l), of which engine paths: $touches_engine"
echo "  processes running out of it: ${busy:-0}"

if [ "${busy:-0}" -gt 0 ] && [ "$touches_engine" -gt 0 ] && [ "$FORCE" != "--force" ]; then
    echo "REFUSED: $busy process(es) are running out of this tree and the incoming diff"
    echo "         touches $touches_engine engine file(s). BoltzGen imports fresh code in every"
    echo "         step subprocess, so the running job would finish on a different engine than"
    echo "         it started on. Wait for it, or pass --force if you know it is past imports."
    printf '%s\n' "$incoming" | grep -E "$ENGINE" | sed 's/^/           /'
    exit 1
fi

ssh -o BatchMode=yes whglx "cd $TREE && git checkout -q --detach origin/wk/mgx-design-accuracy \
    && git log --oneline -1"
