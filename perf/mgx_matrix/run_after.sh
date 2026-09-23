#!/bin/bash
# run_after.sh <run.log to wait on> <run_extra.sh args...>: start the follow-up cells on a card
# only once that card's matrix batch has written its EXIT line, so the two never share a chip.
set -u
cd "$(dirname "$0")/../.."
LOG=$1; shift
until [ -f "$LOG" ] && grep -q '^EXIT=' "$LOG"; do sleep 60; done
exec perf/mgx_matrix/run_extra.sh "$@"
