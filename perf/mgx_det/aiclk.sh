#!/bin/bash
# aiclk.sh <out>: every 30 s, for each chip a mgx-determinism fold has pinned, append
# "<utc> <card> <MHz>" read from the chip's sysfs tt_aiclk (resolved by PCI BDF, as mgx-speed's
# probe_fold.sh does; the UMD id is not the node number on whglx). Reads only, opens no device.
# Exits once no perf/mgx_det run.sh is left.

# One place decides what a valid AICLK is: perf/lib/aiclk.sh, mirroring tt_bio.aiclk.
_L=$(cd "$(dirname "$0")" && pwd); . "${_L%/perf/*}/perf/lib/aiclk.sh" || exit 1
set -u
H=$HOME/wt-mgx-speed/perf/sizegate/campaign
while pgrep -f "perf/mgx_det/run.sh" > /dev/null; do
  for p in $(pgrep -f "tt_bio.main predict $HOME/scratch/mgxdet/"); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q '^TT_BIO_LEASE_HOLDER=worker:mgx-determinism$' || continue
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^TT_VISIBLE_DEVICES=//p'
  done | sort -u | while read -r c; do
    n=$($HOME/env/bin/python -c "import sys;sys.path.insert(0,'$H');import card_health as h;print(h.node_for_bdf(h.bdf_for_umd($c)))" 2>/dev/null)
    echo "$(date -u +%FT%TZ) $c $(aiclk "$n")"
  done >> "$1"
  sleep 30
done
