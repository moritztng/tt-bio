#!/usr/bin/env bash
# Capture the pairformer block boundary at SEVEN depths out of the same single step, so the
# ladder is one function evaluated once rather than seven runs that happen to agree. Blocks 0,
# 23 and 47 are re-captured deliberately: of3t-trunkback already measured those three, so a
# byte-identical re-capture is this row's reproducibility control, and it costs nothing extra
# because one forward and one backward produce all seven boundaries.
#
# The manifest is the one of3t-reference published at 462c1f52, extracted from git into this
# row's own directory. The branch tip has moved on and no longer declares grads_f64_recycles0.pt,
# so reading the manifest from the branch would refuse a bundle that is in fact correct.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkdepth
O=/home/ttuser/of3t_trunkdepth
cd "$W" || exit 1
export PYTHONPATH="/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:/home/ttuser/of3t_gradients/pylibs:$W"
export OMP_NUM_THREADS=8
PY=/home/ttuser/tt-bio-dev/env/bin/python
MAN=$O/MANIFEST_bundle_min_462c1f52.json
git show 462c1f52a09e939298079f066dfe1bf84669fc34:perf/of3t_reference/bundle_min/MANIFEST.json > "$MAN" || exit 1
echo "manifest sha256: $(sha256sum "$MAN")"
echo "=== ladder capture (0,8,16,23,32,40,47) $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_gradients/capture_trunk_boundary.py \
    --blocks 0,8,16,23,32,40,47 --no-dropout --manifest-json "$MAN" \
    --out "$O/cap_ladder" \
    --report perf/of3t_trunkdepth/CAPTURE_LADDER.json
echo "=== capture exit $? $(date -u +%FT%TZ) ==="
echo "CAPLADDER_ALLDONE $(date -u +%FT%TZ)"
