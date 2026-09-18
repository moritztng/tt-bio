#!/usr/bin/env bash
# Is the size-ladder's 256 aa rung FAIL attributable to TT_BIO_DIT_COND_HOIST?
# Runs the gate's own rung command verbatim, hoist off vs on, interleaved, 3 reps each,
# one process per fold exactly as the ladder does. Clock sampled DURING every fold.
set -u
WT=/home/ttuser/.coworker/wt/c13-land-first
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CLK=/sys/class/tenstorrent/tenstorrent!0/tt_aiclk
OUT=$WT/perf/c13_land/ladder_attrib
mkdir -p "$OUT"
cd "$WT" || exit 1
MODEL=${MODEL:-boltz2}
echo "[attrib] model=$MODEL rung=256 steps=6  $(date -Is)"
for rep in 1 2 3; do
  for arm in 0 1; do
    # interior order reversed on even reps so arm order is not confounded with drift
    [ $((rep % 2)) -eq 0 ] && arm=$((1 - arm))
    d=$OUT/${MODEL}_h${arm}_r${rep}
    rm -rf "$d"; mkdir -p "$d"
    ( while :; do cat "$CLK" 2>/dev/null; sleep 0.25; done ) > "$d/clk.txt" &
    SAMP=$!
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c13-land-first \
    TT_BIO_DIT_COND_HOIST=$arm \
      "$PY" -m tt_bio.main predict "$WT/perf/size512/fixtures/cdk2x2_256.yaml" \
        --model "$MODEL" --single_sequence --sampling_steps 6 --diffusion_samples 1 \
        --seed 0 --out_dir "$d" > "$d/fold.log" 2>&1
    rc=$?
    kill "$SAMP" 2>/dev/null; wait "$SAMP" 2>/dev/null
    rt=$("$PY" -c "
import json,glob,sys
f=glob.glob('$d/*/results.json')
if not f: print('NA'); sys.exit()
d=json.load(open(f[0])); e=d[0] if isinstance(d,list) else d
print(e.get('runtime_s','NA'))" 2>/dev/null)
    lo=$(sort -n "$d/clk.txt" 2>/dev/null | head -1); hi=$(sort -n "$d/clk.txt" 2>/dev/null | tail -1)
    n=$(wc -l < "$d/clk.txt" 2>/dev/null)
    echo "rep=$rep hoist=$arm rc=$rc runtime_s=$rt clk=${lo}-${hi} MHz n=$n"
  done
done
echo "[attrib] done $(date -Is)"
