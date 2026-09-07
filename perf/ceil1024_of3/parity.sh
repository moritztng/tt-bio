#!/bin/sh
# Bit-exact A/B of one OpenFold3 fold between two trees, with a mandatory A/A control, on the
# ONE card this workstream was lent.
#
#   TREE_A=/path/to/main TREE_B=/path/to/branch CARD=0 sh parity.sh par_512
#
# Every change on the branch is argued bit-exact -- an in-place op on the same operands in the
# same order, or a partition of an axis nothing reduces over. "Argued" is not "measured". This
# measures it, and bit-exactness is the stronger claim: if no bit moves then every reference
# comparison and every shipped number is unchanged by construction.
#
# The A/A leg runs FIRST and is not optional. Without it a matching A/B proves only that the
# fold is deterministic, and a differing A/B could be run-to-run noise rather than the change.
set -u
rung=$1
TREE_A=${TREE_A:-/home/cust-team/mthuening/ceil1024main}
TREE_B=${TREE_B:-/home/cust-team/mthuening/ceil1024of3}
RUN=${RUN:-/home/cust-team/mthuening/ceil1024/rundir}
OUT=${OUT:-$RUN/parity}
CARD=${CARD:-0}
SEED=${SEED:-42}
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$RUN" || exit 1
mkdir -p "$OUT"

leg() {
  name=$1; tree=$2
  # The resolved module path, not the label, is the evidence for which engine ran.
  eng=$(PYTHONPATH="$tree" "$PY" -c 'import tt_bio, os; print(os.path.dirname(tt_bio.__file__))')
  case "$eng" in "$tree"/*) ;; *) echo "$name REFUSING: $tree resolves to $eng"; return 4 ;; esac
  TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
    TT_BIO_LEASE_HOLDER=worker:ceiling-openfold3-1024 TT_BIO_LEASE_TIMEOUT=20 \
    TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$tree" TT_BIO_SIZE_LIMIT=0 \
    "$PY" -m tt_bio.main predict "msafix_tile/$rung.yaml" --model openfold3 \
    --accelerator tenstorrent --out_dir "$OUT/$name" --override \
    --msa_dir msacache_deep --msa_cache_only --seed "$SEED" > "$OUT/$name.log" 2>&1
  echo "$name rc=$? engine=$eng tree=$(git -C "$tree" rev-parse --short HEAD) $(date -u +%FT%TZ)"
}

leg "${rung}_A1" "$TREE_A"
leg "${rung}_A2" "$TREE_A"
leg "${rung}_B"  "$TREE_B"
echo "--- A/A control (same tree twice): must match, or nothing below means anything ---"
"$PY" "$HERE/../ceiling_of3/parity_cmp.py" "$OUT/${rung}_A1" "$OUT/${rung}_A2"
echo "--- A/B (main vs branch) ---"
"$PY" "$HERE/../ceiling_of3/parity_cmp.py" "$OUT/${rung}_A1" "$OUT/${rung}_B"
