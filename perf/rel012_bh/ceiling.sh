#!/bin/bash
# The p150a ceiling walk at the rel012 integration tree, on pc card 0. Every rung is the real
# BindCraft 2 DESIGN entry (`rung.py` -> `bindcraft2.run_campaign`), two gradient rounds, one
# trajectory. The 768 rung is the axis that crashes on main, so it is a design run here and
# not a fold. 896 is expected to refuse; a refusal naming the size is the supported outcome
# and the rung records its verbatim text.
set -u
wt=/home/moritz/.coworker/wt/rel012-verify-bh
out=$wt/perf/rel012_bh/out/ceiling; mkdir -p "$out"
cd "$wt"
export PYTHONPATH=$wt BCX_BC2=/home/moritz/bcx_shipped/bc2
export JAX_COMPILATION_CACHE_DIR=$wt/perf/rel012_bh/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rel012-verify-bh
PY=/home/moritz/bcx_hostcut_venv/bin/python3
log=$out/ceiling.log
for r in hIL2R:90:512 hIL2R:110:544 hHSA:150:768 hHSA:210:832 hHSA:240:864 hHSA:270:896; do
  IFS=: read -r t b ax <<< "$r"
  tag=a${ax}
  echo "=== $(date -u +%FT%TZ) START $tag $t:$b commit=$(git rev-parse --short HEAD)" >> "$log"
  timeout 3000 $PY -u perf/bgx_size/rung.py --params /home/moritz/bcx_shipped/af2_params \
    --out "$out/$tag" --target "$t" --binder "$b" --rounds 2 --trajectories 1 \
    > "$out/$tag.log" 2>&1
  echo "=== rc=$? $(date -u +%FT%TZ) END $tag" >> "$log"
done
echo "=== CEILING DONE $(date -u +%FT%TZ)" >> "$log"
