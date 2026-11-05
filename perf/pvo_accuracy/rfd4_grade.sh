#!/bin/bash
# RFD4 accuracy grade, pc, no card: the oracle already ran (rfd4_acc.sh wrote the refold cache), so evaluate
# and quality.py are CPU. One call grades every side that has designs.
#   D=~/pvo-accuracy-data/rfd4 bash rfd4_grade.sh
# Layout, the same roots refold.py used (example_id is derived from the path relative to the designs root, so
# evaluate must see the identical root or the cache misses):
#   $D/designs/<side>/        the *_generated.cif, pulled from the box that made them
#   $D/out/<side>/            evaluation_samples/ (refold cache) +
#                            evaluation_metrics/per_example_metrics.csv (where evaluate puts it;
#                            quality.py and rfd4_halves.py rglob for it, so only this script's own
#                            existence check had to learn the subdirectory)
# Sides: gpu-exact, gpu-fast, tt-wh, tt-bh (TT fast = TT exact, one distribution reported as both). The GPU
# designs were lost with their box (11-04), so today: D=... SIDES=tt-bh FLOOR=tt-bh bash rfd4_grade.sh, which
# grades the floor and reports the tt-bh distribution; the GPU comparisons run only for sides that exist.
# Each side also carries `$D/designs/<side>/SOURCE_SHA`, the tt-bio commit its designs were generated on (the
# MODES.md pin, c52ab1702031, unless a side says otherwise). quality.py rebinds a grade to that commit and must
# find it on origin/main; without the file the grade returns REBIND_FAILED and no number is reported. Measured
# 2026-11-04 on a CPU dry run of this exact path: no sha -> REBIND_FAILED, sha -> the grade proceeds.
set -eu
# No `interchain_interactions`: tt-bio does not port that upstream metric (tt_bio/rfd4/evaluate.py
# NOT_PORTED), and asking for the tag is an error, not a shorter CSV. Its three columns are in the
# binder family, so the grade reports them "column absent on both arms" -- which is honest here,
# both arms being tt-bio. Only a missing ORACLE column gates a grade (quality.py:262), and the
# oracle columns all come from the refold cache, so the binder grade still runs on 10 of 13 columns.
D=${D:?}; TAGS=${TAGS:-design,ppi_refolding,hotspot_conditioning}
Q=${Q:-$D/QUALITY-PVO.json}; REPO=${REPO:-/home/moritz/.coworker/wt/pvo-accuracy}
# The harness interpreter is explicit, not whatever PATH yields: tt-bio's venv ships BOTH a real
# python3.10 (what the `tt-bio` console script runs, numpy and all) and a `python3` -> `python`
# symlink to a bare 3.11 with no numpy. Putting that venv's bin on PATH for `tt-bio evaluate`
# therefore breaks every `python3` in this script. Keep them separate.
PY3=${PY3:-/usr/bin/python3}
have() { [ -n "$(find $D/out/$1 -name per_example_metrics.csv 2>/dev/null)" ]; }
SIDES=${SIDES:-"gpu-exact gpu-fast tt-wh"}; FLOOR=${FLOOR:-gpu-exact}

for s in $SIDES; do
  [ -d $D/designs/$s ] || { echo "skip $s: no designs"; continue; }
  have $s ||
    tt-bio evaluate --model rfd4 $D/designs/$s --metrics_tags $TAGS --out_dir $D/out/$s
done

# Floor first, and it is a `resample` floor by name: one arm split in two disjoint halves. With rng=per_design
# the halves are disjoint seed sets, so it bounds seed noise too, but the weaker label is the honest one.
$PY3 $REPO/perf/pvo_accuracy/rfd4_halves.py $D/out/$FLOOR $D/out/_floor
$PY3 $REPO/perf/rfd4/quality.py floor --feature binder --kind resample \
  --a $D/out/_floor/a --b $D/out/_floor/b --source "$FLOOR, $(ls $D/designs/$FLOOR/*.cif | wc -l) designs split in two" --out $Q

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
  $PY3 $REPO/perf/rfd4/quality.py grade --feature binder --label "$1" \
    --tt $D/out/$2 --ref $D/out/$3 --tt-sha "$(sha_of $2)" --ref-sha "$(sha_of $3)" \
    --repo $REPO --out $D/QUALITY-$1.json
  $PY3 $REPO/perf/rfd4/quality.py show --out $D/QUALITY-$1.json
}
for tt in tt-wh tt-bh; do have $tt || continue
  for g in gpu-exact gpu-fast; do have $g && grade $tt-vs-$g $tt $g; done
done
have gpu-fast && have gpu-exact && grade gpu-fast-vs-exact gpu-fast gpu-exact
$PY3 $REPO/perf/rfd4/quality.py show --out $Q
