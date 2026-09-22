#!/usr/bin/env bash
# Both arms of the dead-value release against the SAME float64 reference, on one card.
#
# instrument_a_grad.py writes beside itself, in of3t-equivalence's namespace. That row is
# concluded, so its artifact is restored from git after each arm and only the copy under
# perf/of3t_crop768/out/ is kept.
set -u
CARD=$1
WT=/home/ttuser/.coworker/wt/of3t-crop768
cd "$WT"
for ARM in on off; do
  echo "=== grad arm $ARM $(date -u +%FT%TZ) ==="
  env PYTHONPATH=/home/ttuser/of3t_up/openfold3-0.5.0 TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS=0,"$CARD" TT_BIO_LEASE_HOLDER=worker:of3t-crop768 \
    /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_crop768/grad_ab.py "$ARM" \
      --out perf/of3t_crop768/out/grad_a_"$ARM".json || true
  git checkout -- perf/of3t_equivalence/instrument_a_grad.json 2>/dev/null || true
done
echo "=== grad done $(date -u +%FT%TZ) ==="
