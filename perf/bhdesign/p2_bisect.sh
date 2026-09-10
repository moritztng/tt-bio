#!/bin/bash
# Tighten saprot-35m's DRAM wall. 99999 passes and 131072 throws a 34376517632 B DRAM request
# (131104^2 x 2, the padded L x L attention matrix in bf16) on a 34.22 GB card, so the first
# failure is somewhere just under 131072 and the published cap should not be the loose 99999.
# Both rungs are multiples of 32, which is what the model pads to anyway.
# Waits for the walk-up on card 2 to end rather than queueing behind its own lease.
cd /home/ttuser/.coworker/wt/bh-1536-design-embed-p2 || exit 1
while pgrep -f p2_embed2.sh > /dev/null; do sleep 20; done
export TT_BIO_LEASE_TIMEOUT=1800
python3 perf/bhdesign/ladder.py --model saprot-35m --sizes 114688,126976 \
  --card 2 --holder worker:bh-1536-design-embed-p2 --work perf/bhdesign/work_p2 \
  --timeout 2400 --stop-on-fail --out perf/bhdesign/p2/saprot-35m.jsonl
echo "=== saprot-35m bisect done $(date -u +%H:%M:%SZ) ==="
