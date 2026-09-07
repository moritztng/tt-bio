# Score every rung the ladder recorded as ok. "It returned" is not a PASS.
#
#   sh perf/ceiling_openbind/score_rungs.sh <tree> <out-dir> [rungs-dir]
#
# The rungs dir is separate from the tree because the scorer may run from a harness checkout that
# is not the tree the ladder walked in.
#
# A 1024 fold that completes and hands back a torn structure is worse than the OOM, because it
# looks like success -- this happened on this trunk once already (a TILE reshape across a row
# axis of 249 read back bit-exact and computed wrong, reported at pLDDT 0.90 on a backbone with
# 19-22 A breaks). So the ceiling claim rests on clash fraction, backbone continuity and
# confidence, from perf/wh-correctness/check_structure.py -- the repo's existing detector for
# exactly the plausible-looking wrong answer, with thresholds calibrated against crystal
# structures rather than picked.
set -u
TREE=$1; OUT=$2
RUNGS=${3:-$TREE/rundir/rungs}
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
for r in $(sed -n 's/^RUNG \([a-z0-9_]*\) rc=0 status=ok .*/\1/p' "$OUT/ladder.log"); do
  cif=$(ls "$OUT/$r"/*/predictions/*/*.cif 2>/dev/null | head -1)
  [ -z "$cif" ] && cif=$(find "$OUT/$r" -name '*.cif' | head -1)
  res=$(ls "$OUT/$r"/*/results.json 2>/dev/null | head -1)
  echo "--- $r  $cif"
  ( cd "$TREE" && "$PY" perf/wh-correctness/check_structure.py "$cif" \
      --input "$RUNGS/$r.yaml" ${res:+--conf "$res"} \
      --json "$OUT/$r.signal.json" ) 2>&1 | tail -25
done
