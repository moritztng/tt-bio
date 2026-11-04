#!/bin/bash
# RFD4 accuracy grade, pc, no card: the oracle already ran (rfd4_acc.sh wrote the refold cache), so evaluate
# and quality.py are CPU. One call grades every side that has designs.
#   D=~/pvo-accuracy-data/rfd4 bash rfd4_grade.sh
# Layout, the same roots refold.py used (example_id is derived from the path relative to the designs root, so
# evaluate must see the identical root or the cache misses):
#   $D/designs/<side>/        the *_generated.cif, pulled from the box that made them
#   $D/out/<side>/            evaluation_samples/ (refold cache) + per_example_metrics.csv
# Sides: gpu-exact, gpu-fast, tt-wh (TT fast = TT exact, one distribution reported as both), tt-bh if it lands.
# Each side also carries `$D/designs/<side>/SOURCE_SHA`, the tt-bio commit its designs were generated on (the
# MODES.md pin, c52ab1702031, unless a side says otherwise). quality.py rebinds a grade to that commit and must
# find it on origin/main; without the file the grade returns REBIND_FAILED and no number is reported. Measured
# 2026-11-04 on a CPU dry run of this exact path: no sha -> REBIND_FAILED, sha -> the grade proceeds.
set -eu
D=${D:?}; TAGS=${TAGS:-design,ppi_refolding,hotspot_conditioning,interchain_interactions}
Q=${Q:-$D/QUALITY-PVO.json}; REPO=${REPO:-/home/moritz/.coworker/wt/pvo-accuracy}
SIDES=${SIDES:-"gpu-exact gpu-fast tt-wh"}

for s in $SIDES; do
  [ -d $D/designs/$s ] || { echo "skip $s: no designs"; continue; }
  [ -f $D/out/$s/per_example_metrics.csv ] ||
    tt-bio evaluate --model rfd4 $D/designs/$s --metrics_tags $TAGS --out_dir $D/out/$s
done

# Floor first, and it is a `resample` floor by name: one arm split in two disjoint halves. With rng=per_design
# the halves are disjoint seed sets, so it bounds seed noise too, but the weaker label is the honest one.
python3 $REPO/perf/pvo_accuracy/rfd4_halves.py $D/out/gpu-exact $D/out/_floor
python3 $REPO/perf/rfd4/quality.py floor --feature binder --kind resample \
  --a $D/out/_floor/a --b $D/out/_floor/b --source "gpu-exact, 128 designs split in two" --out $Q

# A grade rewrites the "binder" feature of the doc it reads, so each comparison gets its own file, seeded
# from the floor doc -- without that copy the grade would run with no floor and every bar would be the bare
# p-value.
sha_of() {  # <side>
  f=$D/designs/$1/SOURCE_SHA
  [ -s $f ] || { echo "$1: no $f, the grade would not bind to a commit" >&2; exit 5; }
  tr -d " \t\n" < $f
}
grade() {  # <label> <arm> <reference>
  cp $Q $D/QUALITY-$1.json
  python3 $REPO/perf/rfd4/quality.py grade --feature binder --label "$1" \
    --tt $D/out/$2 --ref $D/out/$3 --tt-sha "$(sha_of $2)" --ref-sha "$(sha_of $3)" \
    --repo $REPO --out $D/QUALITY-$1.json
  python3 $REPO/perf/rfd4/quality.py show --out $D/QUALITY-$1.json
}
grade gpu-fast-vs-exact gpu-fast gpu-exact
[ -d $D/out/tt-wh ] && { grade tt-wh-vs-gpu-exact tt-wh gpu-exact; grade tt-wh-vs-gpu-fast tt-wh gpu-fast; }
