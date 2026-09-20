#!/usr/bin/env bash
# Does TT_BIO_MM_SHORT_M_BW move the Boltz-2 CIF digest between core grids?
#
# The gate's l1-budget arm cannot answer this: it folds protenix-v2, which makes ZERO calls to
# _short_m_proj_config (perf/c14_land/mmshort_firing/summary.json, 26 processes, 0 calls, against
# 576 served on Boltz-2 with the same instrument). So the arm that refused region T is structurally
# blind to this lever and its GATE PASS is not evidence about it.
#
# Four folds, same fixture, same seed, same steps, two grids x flag on/off:
#   ON  native vs ON  8x8   -- the question. Different digests = output depends on the core grid.
#   OFF native vs OFF 8x8   -- the control. These MUST match, or Boltz-2 is grid-dependent without
#                              the lever and the finding is about main, not about the flag.
# 11x10 folds in0_block_w = 6 on (1,512,768)x(768,1536) where 8x8 folds 4, and that shape is served
# 432 times per 6-step fold, so the pair is chosen to separate.
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH="$WT"
OUT="$WT/perf/c14_land/mmshort_grid_digest.txt"
: > "$OUT"
for FLAG in 1 0; do
  for GRID in "11,10" "8,8"; do
    rm -rf /tmp/mmg_out; mkdir -p /tmp/mmg_out
    TT_BIO_MM_SHORT_M_BW=$FLAG TT_BIO_FORCE_GRID=$GRID \
      TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
      timeout 900 "$PY" perf/c14_land/mmkey_census.py --out /tmp/mmg_$FLAG.json \
        --size 512 --steps 6 > /tmp/mmg_run.log 2>&1
    RC=$?
    CIF=$(ls -t "$WT"/perf/c14_land/c14-mmkey-*/out/*.cif 2>/dev/null | head -1)
    MD5=$(md5sum "$CIF" 2>/dev/null | cut -d' ' -f1)
    printf 'flag=%s grid=%s rc=%s md5=%s\n' "$FLAG" "$GRID" "$RC" "${MD5:-NONE}" >> "$OUT"
    cat "$OUT"
  done
done
