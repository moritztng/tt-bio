#!/usr/bin/env bash
# of3t-p10smbw: run a harness on qb2 card 1 with AICLK sampled DURING from sysfs.
#   smbwarm.sh <TAG> <cmd...>
# sysfs (`/sys/class/tenstorrent/tenstorrent!N/tt_aiclk`) rather than `tt-smi -s`, because tt-smi
# OPENS the chip to read telemetry: the sampler in `of3t_p10grad/arm.sh` brought up three cards
# and hung for minutes on this host. sysfs is a read of the class node and touches nothing.
# All four nodes are sampled, so the log shows which card carried the work as well as its clock.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
CARD=${SMBW_CARD:-1}
TAG=${1:?tag}; shift

O=/tmp/of3t/of3t-p10smbw
mkdir -p "$O"
CLK=$O/aiclk_${TAG}.txt
: > "$CLK"
( while true; do
    printf '%s' "$(date -u +%FT%TZ)"
    for n in 0 1 2 3; do
      printf ' %s' "$(cat /sys/class/tenstorrent/tenstorrent\!$n/tt_aiclk 2>/dev/null || echo -)"
    done
    printf '\n'
    sleep 4
  done ) >> "$CLK" &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

S=$(date +%s)
echo "=== smbw $TAG start $(date -u +%FT%TZ) host=$(hostname) card=$CARD head=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
TT_BIO_LEASE_HOLDER=worker:of3t-p10smbw OMP_NUM_THREADS=8 \
"$@" 2>&1 | grep -vE '^\s*$|DEBUG|^Config\{'
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== smbw $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
for n in 0 1 2 3; do
  echo -n "AICLK during, card $n (MHz): "
  awk -v c=$((n+2)) '{print $c}' "$CLK" | grep -E '^[0-9]+$' | sort -n \
    | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
done
exit $rc
