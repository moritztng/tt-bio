#!/bin/bash
# Wait on pc for a free whglx chip, then launch ONE fan and exit. Bounded, and it re-reads the
# quiet-window gate on every iteration.
#
# The problem it solves. whglx has 27 usable chips and a queue of sibling MGX rows, and this
# row twice found 0 of 27 free: leases read `released` while the fds were still open, which is
# the signal that decides. A pass that checks once and gives up loses the chip that frees two
# minutes later; a fan parked on whglx with --retries 3000 would take one, but whglx cannot see
# /home/moritz/.coworker and so cannot honour the quiet window (see launch_if_clear.sh).
#
# So the wait lives here, on the host that CAN see the gate, and it is checked on every pass of
# the loop rather than once at the start -- the window can open while this is waiting.
#
# It stops on its own. A monitor left running past the question it was launched for is its own
# failure mode, so this has a hard deadline and exits after a single successful launch.
#
#   bash perf/mgxaccuracy/watch_and_launch.sh PLAN OUT_JSONL [MAX_MINUTES] [MAX_CONCURRENT] [LINES]
set -u
PLAN=${1:?plan path relative to the repo root}
OUT=${2:?out jsonl path on whglx}
MAX_MIN=${3:-180}
CAP=${4:-1}
LINES=${5:-}
GATE=${MGX_QUIET_WINDOW:-/home/moritz/.coworker/state/mgx/quiet-window}
HERE=$(cd "$(dirname "$0")" && pwd)
deadline=$(( $(date +%s) + MAX_MIN * 60 ))

while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ -e "$GATE" ]; then
        echo "$(date -u +%H:%MZ) quiet window OPEN — waiting, starting nothing"
        sleep 120; continue
    fi
    free=$(ssh -o BatchMode=yes -o ConnectTimeout=30 whglx \
        'cd ~/wt-mgx-design-accuracy && PYTHONPATH=$PWD $HOME/env/bin/python -c "
from perf.mgxscale.job import free_cards
print(\",\".join(str(c) for c in free_cards()))"' 2>/dev/null)
    if [ -n "$free" ]; then
        echo "$(date -u +%H:%MZ) free chips: $free — launching"
        bash "$HERE/launch_if_clear.sh" "$PLAN" "$OUT" "$CAP" "$LINES"
        exit $?
    fi
    echo "$(date -u +%H:%MZ) 0 free chips, retrying"
    sleep 60
done
echo "$(date -u +%H:%MZ) deadline reached after ${MAX_MIN} min with no chip — nothing launched"
exit 2
