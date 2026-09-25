#!/usr/bin/env bash
# Wait for the given pids to exit, then run the rest of the line:  chain.sh <pid,pid,...> <cmd...>
set -u
pids=$1; shift
for p in ${pids//,/ }; do while kill -0 "$p" 2>/dev/null; do sleep 10; done; done
exec "$@"
