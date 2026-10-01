#!/bin/bash
# D8 step 2: the ladder at the commit that carries the measured p150a mode row (cff8269b4),
# memory='auto', two gradient rounds, one trajectory. band.sh already walks 704, 736, 800 and
# 864 at the same commit; this adds the rest of the claimed fast band and the two axes the row
# moved to lean. Every rung rc=0 or the row is wrong.
set -u
wt=/home/moritz/.coworker/wt/rel012-verify-bh
out=$wt/perf/rel012_bh/out/d8; mkdir -p "$out"
log=$out/d8.log
PY=/home/moritz/bcx_hostcut_venv/bin/python3
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rel012-verify-bh
export PYTHONPATH=$wt JAX_COMPILATION_CACHE_DIR=$out/xlacache
cd "$wt"

while ! grep -q "BAND DONE" "$wt/perf/rel012_bh/out/band/band.log" 2>/dev/null; do sleep 20; done

# hIL2R + binder for 576-672 (bcw-bmm's chain read these axes back off the seam),
# hHSA + binder for 768 and 832.
for r in hIL2R:150:576 hIL2R:180:608 hIL2R:210:640 hHSA:60:672 hHSA:150:768 hHSA:210:832; do
  IFS=: read -r t b ax <<< "$r"
  echo "=== $(date -u +%FT%TZ) START a${ax} $t:$b tree=$(git rev-parse --short HEAD)" >> "$log"
  timeout 2400 $PY -u perf/bgx_size/rung.py --params /home/moritz/bcx_shipped/af2_params \
    --out "$out/a${ax}" --target "$t" --binder "$b" --rounds 2 --trajectories 1 \
    > "$out/a${ax}.log" 2>&1
  echo "=== rc=$? $(date -u +%FT%TZ) END a${ax}" >> "$log"
done
echo "=== D8 DONE $(date -u +%FT%TZ)" >> "$log"
