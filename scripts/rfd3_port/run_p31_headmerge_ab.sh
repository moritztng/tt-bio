#!/bin/sh
# p31: alternate the shipped tree and the worktree on one hot card, N times, and report
# the median leg plus the trajectory maxabs between them.
#
# The shipped leg runs against `git archive HEAD` unpacked into a scratch tree, so both
# legs execute the same driver script with the same weights and the same seed and differ
# only in tt_bio/rfd3.py.
#
#   scripts/rfd3_port/run_p31_headmerge_ab.sh [alternations] [batch] [contig]
set -e
WT="$(cd "$(dirname "$0")/../.." && pwd)"
ALT=${1:-3}
BATCH=${2:-1}
CONTIG=${3:-A1-10,230,A31-40}
PY=/home/moritz/tt-bio/env/bin/python3
BASE=/tmp/p31_base
OUT=/tmp/p31_ab
LEASE=worker:tt-bio-rfd3-host-dispatch-p31

rm -rf "$BASE" "$OUT"; mkdir -p "$BASE" "$OUT"
git -C "$WT" archive HEAD tt_bio | tar -x -C "$BASE"

i=1
while [ "$i" -le "$ALT" ]; do
  for LEG in base new; do
    [ "$LEG" = base ] && TREE="$BASE" || TREE="$WT"
    printf '  alternation %s %-4s ' "$i" "$LEG"
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=$LEASE PYTHONPATH="$TREE" \
      "$PY" "$WT/scripts/rfd3_port/p31_headmerge_ab.py" \
      --contig "$CONTIG" --batch "$BATCH" --out "$OUT/$LEG-$i.pt" 2>/dev/null \
      | grep 'ms/step'
  done
  i=$((i + 1))
done

"$PY" - "$OUT" <<'EOF'
import statistics, sys, torch
from pathlib import Path
out = Path(sys.argv[1])
legs = {}
for p in sorted(out.glob("*.pt")):
    leg, _ = p.stem.split("-")
    legs.setdefault(leg, []).append(torch.load(p, weights_only=False))
ref = legs["base"][0]["x"]
maxabs = max((d["x"] - ref).abs().max().item() for v in legs.values() for d in v)
med = {k: statistics.median(m for d in v for m in d["ms"]) for k, v in legs.items()}
d0 = legs["base"][0]
print(f"\n=== p31 head-merge A/B  L={d0['L']} atoms  batch={d0['batch']}  "
      f"steps/leg={d0['steps']}  alternations={len(legs['base'])} ===")
for k in ("base", "new"):
    print(f"{k:<5} median {med[k]:7.2f} ms/step   legs "
          f"{[round(m, 1) for d in legs[k] for m in d['ms']]}")
print(f"speedup            {med['base'] / med['new']:.4f}x  "
      f"({(med['base'] / med['new'] - 1) * 100:+.1f}%)")
print(f"trajectory maxabs vs first base leg: {maxabs:.3e}  "
      f"{'BIT-EXACT' if maxabs == 0.0 else 'NOT BIT-EXACT -- do not ship'}")
EOF
