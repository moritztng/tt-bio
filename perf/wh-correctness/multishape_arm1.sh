#!/bin/bash
# Multi-shape in one process, arm 1: the decisive offline form.
#
# `tt-bio-trace-replay-shape-keyed-multitarget-corruption` was invisible to
# one-fold-per-process testing: state keyed by tensor shape leaked between targets, so a
# fold was only wrong when another shape had gone through the same process first. The only
# test that sees it holds the two arrangements against each other:
#
#   batch  one process, five targets at five different sizes, in one invocation
#   solo   five processes, one target each
#
# Pass = identical CIF sha256 per target. esmfold2-fast because it is single-sequence (no
# MSA-server variance to confound the comparison) and folds every one of these sizes on
# this box, measured 7/7 by the size axis. The seed is explicit so a seed difference cannot
# masquerade as corruption.
#
# Runs on UF-EV-A13-GWH02 against one free card. Not on the production pool's cards: check
# `sudo lsof /dev/tenstorrent/*` immediately before launching and pass a free one as $1.
set -u
CARD="${1:?usage: multishape_arm1.sh <umd_device_id>}"
ROOT=/home/cust-team/mthuening/whcorr_multishape
REPO=/home/cust-team/mthuening/whbase/tt-bio
PY=/home/cust-team/mthuening/tt-bio/env/bin/python
SIZES="128 298 384 512 640"

mkdir -p "$ROOT/targets" "$ROOT/out"
cd "$REPO" || exit 1

# The fleet's own size fixture: CDK2 (1HCL) tiled and truncated, the same construction
# perf/size512/fixtures uses, so these sizes are comparable with what wh-perf-* measures.
PYTHONPATH=. "$PY" - "$ROOT/targets" $SIZES <<'EOF'
import sys, pathlib
CDK2 = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEF"
        "LHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVV"
        "TLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQ"
        "DFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")
d = pathlib.Path(sys.argv[1])
for n in (int(x) for x in sys.argv[2:]):
    seq = (CDK2 * (n // len(CDK2) + 1))[:n]
    (d / f"cdk2_{n}.yaml").write_text(f"sequences:\n  - protein: {{id: A, sequence: {seq}}}\n")
EOF

run() {  # run <out_subdir> <input path>
  PYTHONPATH=. "$PY" -c 'from tt_bio.main import cli; cli()' predict "$2" \
      --model esmfold2-fast --seed 0 --device_ids "$CARD" --host_threads 16 \
      --out_dir "$ROOT/out/$1" --debug
}

echo "=== batch: one process, five targets ==="
run batch "$ROOT/targets"

echo "=== solo: five processes, one target each ==="
for n in $SIZES; do
  echo "--- solo $n ---"
  run "solo_$n" "$ROOT/targets/cdk2_$n.yaml"
done

echo "=== verdict ==="
for n in $SIZES; do
  b=$(find "$ROOT/out/batch" -name "cdk2_${n}*.cif" -o -name "*cdk2_${n}*.cif" | head -1)
  s=$(find "$ROOT/out/solo_$n" -name "*.cif" | head -1)
  hb=$( [ -n "$b" ] && sha256sum "$b" | cut -c1-16 || echo MISSING )
  hs=$( [ -n "$s" ] && sha256sum "$s" | cut -c1-16 || echo MISSING )
  v=$( [ "$hb" = "$hs" ] && [ "$hb" != MISSING ] && echo IDENTICAL || echo "**DIFFERS**" )
  echo "$n  batch=$hb  solo=$hs  $v"
done
