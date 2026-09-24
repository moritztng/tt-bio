#!/bin/bash
# Launch the whglx design fan ONLY if the MGX quiet window is closed.
#
# The quiet window (orchestrator note 2026-09-23 16:10 UTC) forbids STARTING any whglx job or
# CPU job while /home/moritz/.coworker/state/mgx/quiet-window exists; jobs already running
# finish. The note says to check before every launch "including inside your own chain or queue
# scripts", and that is the problem this script solves: the fan runs ON WHGLX, which has no
# /home/moritz/.coworker at all, so a fan cannot check the gate itself. Verified -- neither
# ~/.coworker nor /home/moritz/.coworker exists there.
#
# So the gate has to be enforced on pc, at launch time, which means no long-lived autonomous
# launcher may sit on whglx across a window it cannot see. This row's fan was stopped for that
# reason; its running jobs were left alone.
#
#   bash perf/mgxaccuracy/launch_if_clear.sh PLAN_PATH OUT_JSONL [MAX_CONCURRENT] [LINES]
#
# LINES picks plan entries by 1-based index. Use it when the window could open while the fan
# is mid-job: the fan exits after the entries it was given instead of sitting on whglx queued
# for the rest, which is the one case where the per-launch gate above is not enough.
set -u
# Overridable ONLY so the refusal path can be tested without creating the real file, which
# would stop every other MGX row.
GATE=${MGX_QUIET_WINDOW:-/home/moritz/.coworker/state/mgx/quiet-window}
PLAN=${1:?plan path relative to the repo root}
OUT=${2:?out jsonl path on whglx}
CAP=${3:-3}
LINES=${4:-}
# --max-concurrent caps THIS fan; the brief's 3-chip grant is per ROW, and on 2026-09-24 at
# 07:17Z a third fan launched with cap 1 while two fans already held three chips took the row
# to four. fan.py --row-cap counts the row's chips from the lease dir, the only state the fans
# share, so enforcing it here makes that arithmetic unrepeatable from any caller.
ROW_CAP=${MGX_ROW_CAP:-3}

# OUT is single-quoted inside the remote command, so it is NOT expanded on whglx: a
# $HOME-relative path arrives literally and fan.py creates a directory called '$HOME' inside
# the checkout. Caught by passing one.
case "$OUT" in
    /*) ;;
    *) echo "REFUSED: OUT must be an absolute path on whglx (got '$OUT') — a \$HOME-relative"
       echo "         one is not expanded there and lands as a literal directory in the repo."
       exit 1;;
esac

if [ -e "$GATE" ]; then
    echo "REFUSED: the MGX quiet window is OPEN, so nothing new starts. Contents:"
    sed 's/^/    /' "$GATE"
    echo "Re-run this script after it closes. Jobs already running are unaffected."
    exit 1
fi
# The brief's 2026-09-23 17:5x note: merge main before the next device run. The launcher is
# where that has to be enforced, because the tree a job runs on is decided here and nowhere
# else -- and this row's whglx checkout is deliberately held BEHIND main while two arms are
# mid-run, so "the branch is merged on pc" says nothing about what whglx would execute.
# Checked against the checkout that will actually run the job, not against this worktree.
REMOTE_TREE=${MGX_REMOTE_TREE:-\$HOME/wt-mgx-design-accuracy}
#
# OVERRIDE, deliberately narrow. A CATCHER has to be attributable to main and that is what the
# guard protects. A SIZE LADDER has the opposite requirement: its rungs are comparable only if
# they share a tree, and the tree its earlier rungs were measured on is by then behind main. So
# MGX_ALLOW_BEHIND_MAIN takes the path of a committed plan file whose header states which banked
# cells the run must share a tree with, the reason is echoed into the launch output, and a bare
# =1 is refused -- an override without a written reason is how a catcher ends up on a stale tree.
behind=$(ssh -o BatchMode=yes -o ConnectTimeout=30 whglx \
    "cd $REMOTE_TREE && git fetch -q origin main 2>/dev/null; \
     git merge-base --is-ancestor origin/main HEAD && echo ok || echo behind" 2>/dev/null)
ALLOW_BEHIND=${MGX_ALLOW_BEHIND_MAIN:-}
if [ -n "$ALLOW_BEHIND" ] && [ "$behind" != "ok" ]; then
    if [ ! -f "$ALLOW_BEHIND" ]; then
        echo "REFUSED: MGX_ALLOW_BEHIND_MAIN must be the path of a committed plan file stating"
        echo "         which banked cells this run shares a tree with (got '$ALLOW_BEHIND')."
        exit 1
    fi
    echo "behind-main override accepted, reason from $ALLOW_BEHIND:"
    grep -m1 -n "ONE TREE" "$ALLOW_BEHIND" | sed 's/^/    /'
    echo "    remote tree: $(ssh -o BatchMode=yes -o ConnectTimeout=30 whglx "cd $REMOTE_TREE && git rev-parse --short HEAD" 2>/dev/null)"
    behind=ok
fi
if [ "$behind" != "ok" ]; then
    echo "REFUSED: the whglx checkout does not contain origin/main (got '${behind:-no answer}')."
    echo "         A device run from it is attributable to a tree that is not main. Update the"
    echo "         checkout first -- but NOT while a measurement is mid-run, since BoltzGen"
    echo "         imports fresh code in every step subprocess."
    exit 1
fi

# The plan is read on WHGLX, from a tree the launcher does not update -- it fetches and
# never checks out, so a plan committed after that tree was created is simply absent there.
# Caught by launching one: the launcher printed "launched" and the fan died one second later
# on FileNotFoundError, which looks from here exactly like a successful start.
if ! ssh -o BatchMode=yes -o ConnectTimeout=30 whglx "test -f $REMOTE_TREE/$PLAN" 2>/dev/null
then
    echo "REFUSED: $PLAN does not exist in $REMOTE_TREE on whglx."
    echo "         The launcher fetches but does not check out, so a plan committed after that"
    echo "         tree was last updated is missing there. Check the tree out, then re-run."
    exit 1
fi

echo "quiet window closed ($GATE absent) -- launching"

# The launch must use the SAME tree the guard above checked, or the guard checks one
# checkout and the job runs from another. MGX_REMOTE_TREE selects it; the default is this
# row's main checkout, and the catcher passes ~/wt-mgx-catcher, which is a second worktree at
# the merged head so the two 1536 arms keep the 7cf87b844 tree they started on.
ssh -o BatchMode=yes whglx "cd $REMOTE_TREE \
  && git fetch -q origin wk/mgx-design-accuracy \
  && export TT_BIO_LEASE_DIR=\$HOME/leases PYTHONPATH=\$PWD LADDER_PY=\$HOME/env/bin/python \
  && setsid nohup \$HOME/env/bin/python -u perf/mgxscale/fan.py \
       --plan '$PLAN' --out '$OUT' --holder worker:mgx-design-accuracy \
       --max-concurrent $CAP --row-cap $ROW_CAP \
       --work \$HOME/mgxacc-work --wait-s 10 --retries 3000 \
       ${LINES:+--lines $LINES} \
       > \$HOME/mgxacc-work/fan_\$(date +%H%M%S).log 2>&1 < /dev/null &
     sleep 8; echo launched" 2>&1 | tail -3
