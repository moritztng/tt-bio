#!/usr/bin/env bash
# of3t-p10exact: run one python arm on qb2 card 1 with an AICLK sampler beside it.
#
#   run.sh <TAG> <python args...>
#
# The clock comes from `/sys/class/tenstorrent/tenstorrent!<card>/tt_aiclk`, not `tt-smi -s`.
# tt-smi snapshots every chip on the host, so one busy or wedged sibling hangs the sampler and
# the arm records no clock at all; the sysfs attribute is per-card and returns instantly.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-p10exact
O=/home/ttuser/of3t_p10exact
CARD=1
cd "$W"
mkdir -p "$O"
TAG=${1:?usage: run.sh TAG python-args...}; shift

CLK=$O/aiclk_${TAG}.txt
: > "$CLK"
( while true; do cat "/sys/class/tenstorrent/tenstorrent!$CARD/tt_aiclk" >> "$CLK" 2>/dev/null; sleep 2; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

source /home/ttuser/tt-bio-dev/env/bin/activate
unset TT_MESH_GRAPH_DESC_PATH
S=$(date +%s)
echo "=== $TAG start $(date -u +%FT%TZ) $(hostname) card $CARD ==="
env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-p10exact \
    TT_BIO_SOFTMAX_BW_RENORM=1 OMP_NUM_THREADS=8 PYTHONPATH="$W" \
    timeout 5400 python3 "$@" 2>&1 | tee "$O/raw_${TAG}.log"
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill $SAMPLER 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
python3 - "$CLK" <<'PY'
import sys
s = sorted(int(l) for l in open(sys.argv[1]) if l.strip().isdigit())
print("AICLK sampled DURING: n=%d min=%d median=%d max=%d" % (len(s), s[0], s[len(s)//2], s[-1]) if s else "AICLK: no samples")
PY
exit "$rc"
