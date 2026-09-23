#!/bin/bash
# after.sh <pid> <run.sh args...>: start run.sh once process <pid> has exited, so a queued
# repeat set never adds a chip to the ones this row already holds.
set -u
P=$1; shift
while kill -0 "$P" 2>/dev/null; do sleep 60; done
exec "$(dirname "$0")/run.sh" "$@"
