#!/usr/bin/env bash
# Perf leg of the post-b2z2 full gate, qb2 card 1. Takes ONE benchlock hold for both steps
# (see perf_steps.sh) rather than letting them race each other or a co-tenant.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
exec /home/ttuser/.coworker/scripts/benchlock.sh tt-bio-full-gate-post-b2z2/perf -- "$DIR/perf_steps.sh"
