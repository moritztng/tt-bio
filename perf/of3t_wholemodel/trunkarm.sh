#!/usr/bin/env bash
# The trunk's fourth arm. `of3t-apbgrad` measured shipped and renorm at this boundary; nobody
# has asked the trunk whether the host float64 softmax reaches it at all. The wrapper answers
# that as a counter rather than as an inference from where the call sites are.
#   trunkarm.sh <shipped|renorm|renormf64>
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
O=/home/ttuser/of3t_wholemodel
mkdir -p "$O"
B=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
ARM=$1
case "$ARM" in
  shipped)   ;;
  renorm)    export TT_BIO_SOFTMAX_BW_RENORM=1 ;;
  renormf64) export TT_BIO_SOFTMAX_BW_RENORM=1; export TT_BIO_HOST_F64_SOFTMAX_AB=all ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-wholemodel
export PYTHONPATH="$W/perf/of3t_tape:$W"
S=$(date +%s)
echo "=== trunk $ARM start $(date -u +%FT%TZ) card $CARD ==="
timeout 2400 "$PY" perf/of3t_wholemodel/armrun.py perf/of3t_bwdaccum/dev_cot.py --lever none \
  --cot-out "$O/trunk_devcot_${ARM}_c64.pt" --ln-capture 0,12,24,36,47 \
  --ln-out "$O/trunk_ln_${ARM}_c64.pt" \
  --boundary "$B" --cap-last "$C" --out "$O/trunk_${ARM}_c64.pt" \
  --report "perf/of3t_wholemodel/TRUNK_ARM_${ARM}_c64.json" --arm flipped
rc=$?
E=$(date +%s)
echo "=== trunk $ARM exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
exit $rc
