#!/bin/bash
cd /home/moritz/.coworker/wt/japanfold-wh-correctness-close/perf/wh-correctness
PY=/home/moritz/tt-bio/env/bin/python3
for c in nomsa_128_opendde nomsa_128_opendde-abag nomsa_64_opendde nomsa_64_opendde-abag; do
  echo "=== $c ==="
  "$PY" jf_cell.py --cell "$c" --kind predict --expect ok \
    --payload /tmp/odde_nomsa/$c.json --input /tmp/odde_nomsa/$c.yaml \
    --out results/odde_nomsa.jsonl --artifacts results/artifacts_nomsa --deadline 1800
done
