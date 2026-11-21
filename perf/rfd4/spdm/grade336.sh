#!/bin/bash
# grade336.sh OUT SEEDS ARM...  : designs at the gapboard 336-token shape (perf/dsfix/fixtures/rfd4_S336.yaml,
# recipe thinking_medium, --fast) for each lever arm at every seed 0..SEEDS, one design a seed, on one chip.
# Run under run_cell.sh, which sets PYTHONPATH, TT_VISIBLE_DEVICES and samples AICLK. An arm is `base` (the
# shipped fast set) or `+a,+b` added to it. The CPU grade leg is card-free: perf/rfd4/spdm/grade336_read.py OUT.
set -u
OUT=$1 N=$2; shift 2
WT=$(cd "$(dirname "$0")/../../.." && pwd)
W=$(cd "$WT" && python -c "from tt_bio.rfd4.engine import _resolve_weights as r; print(r('nvidia/RFD4-Proteina'))") || exit 2
SPEC=$WT/perf/dsfix/fixtures/rfd4_S336.yaml
BASE=$(cd "$WT" && python -c "from tt_bio.rfd4.trunk import FAST_LEVERS as F; print(','.join(sorted(F)))") || exit 2
mkdir -p "$OUT"
for arm in "$@"; do
  L=$BASE; [ "$arm" = base ] || L="$BASE,${arm//+/}"
  echo "$(date -u +%FT%TZ) arm $arm seeds 0..$N" >&2
  rm -rf "$OUT/$arm"  # a retried cell must not meet its killed attempt's cifs: write_cif never overwrites
  TT_BIO_LEVERS=$L python "$WT/perf/rfd4/e2e/design.py" --spec "$SPEC" --weights "$W" --fast --rng upstream \
    --seeds $(seq 0 "$N") --out "$OUT/$arm" > "$OUT/$arm.log" 2>&1
  echo "$(date -u +%FT%TZ) arm $arm rc=$?" >> "$OUT/done"
done
touch "$OUT/designs.done"
