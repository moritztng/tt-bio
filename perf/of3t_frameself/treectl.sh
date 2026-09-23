#!/usr/bin/env bash
# of3t-frameself: are the two `of3pkg043` trees the campaign uses actually the same tree?
#
# of3t-modelframe's capture ran under /home/ttuser/of3t-campaign-refs/of3pkg043 (on qb1) and
# of3t-twoside's replay under /home/ttuser/of3t_refprec/of3pkg043 (on qb2). Both are called
# "upstream 0.4.3". If they differ in pairformer.py the whole D242 gap is explained without any
# further experiment, so this is checked before anything expensive is read.
#
# The digest is over CONTENTS ONLY. `find | xargs sha256sum | sha256sum` also hashes the paths,
# which differ by the root, and reports two identical trees as different -- the first cut of
# this control did exactly that.
set -uo pipefail
A=/home/ttuser/of3t-campaign-refs/of3pkg043
B=/home/ttuser/of3t_refprec/of3pkg043
ha=$(cd "$A" && find . -name '*.py' -type f | sort | xargs cat | sha256sum | cut -d' ' -f1)
hb=$(cd "$B" && find . -name '*.py' -type f | sort | xargs cat | sha256sum | cut -d' ' -f1)
na=$(cd "$A" && find . -name '*.py' -type f | wc -l)
nb=$(cd "$B" && find . -name '*.py' -type f | wc -l)
pa=$(sha256sum "$A/openfold3/core/model/latent/pairformer.py" | cut -d' ' -f1)
pb=$(sha256sum "$B/openfold3/core/model/latent/pairformer.py" | cut -d' ' -f1)
extra=$(diff -rq "$A" "$B" 2>&1 | grep -v __pycache__ | sed 's/"/'"'"'/g' | paste -sd'|' -)
cat <<JSON
{
 "what": "the capture's tree against the replay's tree, contents only",
 "host": "$(hostname)",
 "row": "of3t-frameself", "defect": "D242", "device_involved": false,
 "why_no_aiclk": "CPU only, no Tenstorrent device is opened",
 "capture_tree": {"path": "$A", "used_by": "of3t-modelframe/capture.sh, on qb1",
                  "n_py_files": $na, "sha256_of_concatenated_py": "$ha",
                  "pairformer_py_sha256": "$pa"},
 "replay_tree":  {"path": "$B", "used_by": "of3t-twoside/arms.sh -> ref_grad.py, on qb2",
                  "n_py_files": $nb, "sha256_of_concatenated_py": "$hb",
                  "pairformer_py_sha256": "$pb"},
 "identical_py_contents": $([ "$ha" = "$hb" ] && echo true || echo false),
 "pairformer_identical": $([ "$pa" = "$pb" ] && echo true || echo false),
 "non_py_differences": "$extra",
 "verdict": "$([ "$ha" = "$hb" ] && echo 'the two trees are the same code; a tree difference is NOT the cause of D242' || echo 'THE TREES DIFFER -- see non_py_differences')"
}
JSON
