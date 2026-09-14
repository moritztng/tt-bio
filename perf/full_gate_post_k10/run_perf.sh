#!/usr/bin/env bash
# Perf + digest leg of the post-K10 full gate, qb2 card $GATE_CARD (default 0). ONE benchlock hold for all three steps
# (see perf_steps.sh) rather than letting them race each other or a co-tenant on the board pair.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
exec /home/ttuser/.coworker/scripts/benchlock.sh tt-bio-full-gate-post-k10/perf -- "$DIR/perf_steps.sh"
