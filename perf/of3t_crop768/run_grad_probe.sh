#!/usr/bin/env bash
set -u
CARD=$1
WT=/home/ttuser/.coworker/wt/of3t-crop768
cd "$WT"
for ARM in on off; do
  echo "=== probe arm $ARM $(date -u +%FT%TZ) ==="
  env PYTHONPATH=/home/ttuser/of3t_up/openfold3-0.5.0 TT_VISIBLE_DEVICES="$CARD" \
    TT_BIO_LEASE_CARDS=0,"$CARD" TT_BIO_LEASE_HOLDER=worker:of3t-crop768 \
    /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_crop768/grad_probe.py "$ARM" \
      --out perf/of3t_crop768/out/grad_probe_"$ARM".json 2>&1 | grep -E "^\[probe\]|^worst|^VERDICT|Error|Traceback" || true
  git checkout -- perf/of3t_equivalence/instrument_a_grad.json 2>/dev/null || true
done
echo "=== probe done $(date -u +%FT%TZ) ==="
