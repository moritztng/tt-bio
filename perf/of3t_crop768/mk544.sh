#!/usr/bin/env bash
# Regenerate the 544 rung fixture. perf/size512/fixtures/ is not this row is namespace, so the
# generated cdk2x2_544.{yaml,a3m} are left untracked and rebuilt here instead of committed.
# The generator carries its own L=298 self-check and keeps MSA depth at 35.
set -eu
cd /home/ttuser/.coworker/wt/of3t-crop768
exec /home/ttuser/tt-bio-dev/env/bin/python3 perf/size512/build_sweep_fixtures.py 544
