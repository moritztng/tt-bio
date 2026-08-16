#!/bin/bash
# Multi-shape in one process, arm 1 — the offline form the sweep never got to run.
#
# `tt-bio-trace-replay-shape-keyed-multitarget-corruption` was invisible to
# one-fold-per-process testing: state keyed on tensor shape leaked between targets, so a
# fold was only wrong when a different shape had gone through the same process first. The
# only arrangement that sees it holds two against each other:
#
#   batch  one process, six folds: five sizes ascending, then the first size AGAIN
#   solo   five processes, one size each, fresh process every time
#
# Pass = identical CIF sha256 per size, on both comparisons:
#   batch 128 (first) == batch 128 (repeat)   -> not process state
#   batch N == solo N, all five               -> not cross-process either
# The trailing repeat is what makes this sharper than a plain batch-vs-solo diff: if it
# disagrees with the first 128 in its OWN process, that is state, with no cross-process
# variable left to blame. Sizes are all distinct so no swap can hide behind an equal
# residue count.
#
# protenix-v2 (§8): the trace-capable path, and the model whose module-scope cache already
# caused one live incident (`meshdevice-remote-only-stale-cache`). --single_sequence is
# mandatory, not a speed choice: without it protenix reaches out to api.colabfold.com and a
# remote MSA search is the one input that could differ between two arms on its own, which
# would make every hash comparison below meaningless. Explicit
# seed, same flags on both arms: bit-exactness is a fair bar only because every one of
# those is held fixed. If any differ, the run is invalid, not the service.
#
# Blackhole host: the defect class is Python-level module-scope state keyed on shape, which
# is architecture-independent. This proves the state machinery, NOT the Wormhole numerics.
set -u
CARD="${1:-0}"
ROOT="${2:-/tmp/whcorr_multishape}"
REPO=/home/moritz/.coworker/wt/japanfold-wh-correctness-close
PY=/home/moritz/tt-bio/env/bin/python3
MODEL="${MODEL:-protenix-v2}"

mkdir -p "$ROOT/targets" "$ROOT/out"
cd "$REPO" || exit 1

# Numeric filename prefixes pin the in-process order: 128, 256, 384, 512, 640, then 128
# again. Without them the repeat sorts next to the original and stops being a trailing one.
PYTHONPATH=. "$PY" - "$ROOT/targets" <<'EOF'
import pathlib, sys
CDK2 = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEF"
        "LHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVV"
        "TLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQ"
        "DFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")
d = pathlib.Path(sys.argv[1])
plan = [("01", 128), ("02", 256), ("03", 384), ("04", 512), ("05", 640), ("06", 128)]
for tag, n in plan:
    seq = (CDK2 * (n // len(CDK2) + 1))[:n]
    (d / f"{tag}_cdk2_{n}.yaml").write_text(
        f"sequences:\n  - protein: {{id: A, sequence: {seq}}}\n")
EOF
# The repeat must be byte-identical input to the original, or the comparison is vacuous.
cmp -s "$ROOT/targets/01_cdk2_128.yaml" "$ROOT/targets/06_cdk2_128.yaml" \
  || { echo "FATAL: repeat input differs from the original"; exit 1; }

run() {  # run <out_subdir> <input path>
  TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_HOLDER=worker:japanfold-wh-correctness-close \
  PYTHONPATH=. "$PY" -m tt_bio.main predict "$2" \
      --model "$MODEL" --seed 0 --single_sequence --out_dir "$ROOT/out/$1"
}

echo "=== batch: one process, six folds (128 256 384 512 640 128) ==="
run batch "$ROOT/targets"

echo "=== solo: five processes, one size each ==="
for t in 01_cdk2_128 02_cdk2_256 03_cdk2_384 04_cdk2_512 05_cdk2_640; do
  echo "--- solo $t ---"
  mkdir -p "$ROOT/targets_solo/$t" && cp "$ROOT/targets/$t.yaml" "$ROOT/targets_solo/$t/"
  run "solo_$t" "$ROOT/targets_solo/$t/$t.yaml"
done

h() { local f; f=$(find "$1" -name "*$2*.cif" 2>/dev/null | sort | head -1)
      [ -n "$f" ] && sha256sum "$f" | cut -c1-16 || echo MISSING; }

echo "=== verdict: batch vs solo ==="
for t in 01_cdk2_128 02_cdk2_256 03_cdk2_384 04_cdk2_512 05_cdk2_640; do
  hb=$(h "$ROOT/out/batch" "$t"); hs=$(h "$ROOT/out/solo_$t" "$t")
  v=$( [ "$hb" = "$hs" ] && [ "$hb" != MISSING ] && echo IDENTICAL || echo "**DIFFERS**" )
  echo "$t  batch=$hb  solo=$hs  $v"
done

echo "=== verdict: the trailing repeat, same process ==="
h1=$(h "$ROOT/out/batch" "01_cdk2_128"); h6=$(h "$ROOT/out/batch" "06_cdk2_128")
v=$( [ "$h1" = "$h6" ] && [ "$h1" != MISSING ] && echo IDENTICAL || echo "**DIFFERS**" )
echo "128 first=$h1  128 repeat=$h6  $v"
