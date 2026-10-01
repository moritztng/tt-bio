#!/bin/bash
# 768 and 832 refuse at the shipped default while 864 and 896 serve, and both refusals repeated
# exactly. So the question is no longer "where is the ceiling" but "which axes in the fast band
# does this board actually hold". This walks the rest of that band at memory='auto': 704, 736,
# 800, and 864 again to see whether the one that served repeats too.
set -u
wt=/home/moritz/.coworker/wt/rel012-verify-bh
out=$wt/perf/rel012_bh/out/band; mkdir -p "$out"
log=$out/band.log
PY=/home/moritz/bcx_hostcut_venv/bin/python3
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rel012-verify-bh
export PYTHONPATH=$wt JAX_COMPILATION_CACHE_DIR=$out/xlacache
cd "$wt"

while ! grep -q "ROUND DONE" "$wt/perf/rel012_bh/out/round/round.log" 2>/dev/null; do sleep 20; done

for r in 90:704 110:736 180:800 240:864; do
  IFS=: read -r b ax <<< "$r"
  echo "=== $(date -u +%FT%TZ) START a${ax} binder=$b tree=$(git rev-parse --short HEAD)" >> "$log"
  timeout 2400 $PY -u perf/bgx_size/rung.py --params /home/moritz/bcx_shipped/af2_params \
    --out "$out/a${ax}" --target hHSA --binder "$b" --rounds 2 --trajectories 1 \
    > "$out/a${ax}.log" 2>&1
  echo "=== rc=$? $(date -u +%FT%TZ) END a${ax}" >> "$log"
done
echo "=== BAND DONE $(date -u +%FT%TZ)" >> "$log"
