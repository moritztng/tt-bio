#!/bin/bash
# Scout-only: run one probe/harness under one ttnn version on card 1.
#   scout_run_leg.sh <68|75> <script> [args...]
# 0.68 runs under the host's stock SFPI 7.35.3; 0.75 needs SFPI 7.67.0, bind-mounted over
# /opt/tenstorrent/sfpi inside a private mount namespace so the shared host install and the
# concurrently-running release gate are untouched.
set -u
VER="$1"; SCRIPT="$2"; shift 2; ARGS="$*"
WT=/home/ttuser/.coworker/wt/tt-bio-ttnn-0-75-perf-exploit-p2
COMMON="PYTHONNOUSERSITE=1 TT_VISIBLE_DEVICES=${SCOUT_CARD:-1} TT_BIO_LEASE_HOLDER=worker:tt-bio-ttnn-0-75-perf-exploit-p2"

if [ "$VER" = "68" ]; then
  cd "$WT" || exit 1
  exec env $COMMON TT_METAL_CACHE=/home/ttuser/.coworker/scout-cache68 \
    /home/ttuser/.coworker/scout-venvs/v68/bin/python "$SCRIPT" $ARGS
else
  exec unshare --map-root-user --mount bash -c "
    mount --bind /home/ttuser/.coworker/scout-sfpi/extracted/sfpi /opt/tenstorrent/sfpi || exit 1
    cd $WT || exit 1
    exec env $COMMON TT_METAL_CACHE=/home/ttuser/.coworker/scout-cache75 \
      /home/ttuser/.coworker/scout-venvs/v75/bin/python $SCRIPT $ARGS
  "
fi
