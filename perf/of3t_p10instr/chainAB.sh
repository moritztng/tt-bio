#!/bin/bash
# of3t-p10instr: score the pre-registered A/B arms (PREREGISTRATION.md, eefcb7234), in the order
# it lists. Waits for the chain whose output it needs by reading that chain's log, not by pgrep.
#   chainAB.sh two     card 2, after chainN (N3b done): base, TF7, B2trunk, I_TFs1
#   chainAB.sh twelve  card 1, after chainT (CHAINT DONE): base, I_D12, X12b, I_D12s1
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
L=/home/ttuser/of3t_p10instr/logs
R=/home/ttuser/of3t_p10trainout/runs
C=/home/ttuser/of3t_p10trainout/corpus
cd "$W" || exit 1
case $1 in
  two)    card=2; gate="N3b done"; tag=AB2
          arms=(--adapter base --adapter $R/TF7/adapter-00000001.safetensors
                --adapter $R/B2trunk/adapter-00000001.safetensors
                --adapter $R/I_TFs1/adapter-00000001.safetensors) ;;
  twelve) card=1; gate="CHAINT DONE"; tag=AB12
          arms=(--adapter base --adapter $R/I_D12/adapter-00000011.safetensors
                --adapter $R/X12b/adapter-00000011.safetensors
                --adapter $R/I_D12s1/adapter-00000011.safetensors) ;;
esac
log=$L/chainN.log; [ "$1" = twelve ] && log=$L/chainT.log
until grep -q "$gate" "$log" 2>/dev/null; do sleep 30; done
echo "=== $tag start $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) card $card" >> "$L/chainAB.log"
env TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:of3t-p10instr \
    timeout 3600 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_p10instr/evalnoise.py \
    --train-corpus $C/train12 --eval-corpus $C/val \
    --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --seeds 20260926,0-31 --out perf/of3t_p10instr/out/$tag.json "${arms[@]}" > "$L/$tag.log" 2>&1
echo "=== $tag done $(date -u +%FT%TZ) rc=$?" >> "$L/chainAB.log"
