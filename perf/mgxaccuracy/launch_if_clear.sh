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
#   bash perf/mgxaccuracy/launch_if_clear.sh PLAN_PATH OUT_JSONL [MAX_CONCURRENT]
set -u
# Overridable ONLY so the refusal path can be tested without creating the real file, which
# would stop every other MGX row.
GATE=${MGX_QUIET_WINDOW:-/home/moritz/.coworker/state/mgx/quiet-window}
PLAN=${1:?plan path relative to the repo root}
OUT=${2:?out jsonl path on whglx}
CAP=${3:-3}

if [ -e "$GATE" ]; then
    echo "REFUSED: the MGX quiet window is OPEN, so nothing new starts. Contents:"
    sed 's/^/    /' "$GATE"
    echo "Re-run this script after it closes. Jobs already running are unaffected."
    exit 1
fi
echo "quiet window closed ($GATE absent) -- launching"

ssh -o BatchMode=yes whglx "cd ~/wt-mgx-design-accuracy \
  && git fetch -q origin wk/mgx-design-accuracy \
  && export TT_BIO_LEASE_DIR=\$HOME/leases PYTHONPATH=\$PWD LADDER_PY=\$HOME/env/bin/python \
  && setsid nohup \$HOME/env/bin/python -u perf/mgxscale/fan.py \
       --plan '$PLAN' --out '$OUT' --holder worker:mgx-design-accuracy \
       --max-concurrent $CAP --work \$HOME/mgxacc-work --wait-s 10 --retries 3000 \
       > \$HOME/mgxacc-work/fan_\$(date +%H%M%S).log 2>&1 < /dev/null &
     sleep 8; echo launched" 2>&1 | tail -3
