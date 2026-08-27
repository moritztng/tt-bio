#!/usr/bin/env bash
# Fold the same target from TWO checkouts and compare, without accidentally comparing one
# checkout against itself.
#
# The trap this exists to prevent: `python -m` puts the CURRENT DIRECTORY at sys.path[0], ahead
# of PYTHONPATH. So an A/B that varies only PYTHONPATH, run from inside one of the two trees,
# silently imports THAT tree in both arms. Both arms then agree bit-for-bit -- which is exactly
# the answer "my change is inert" is hoping for, so nothing looks wrong. It cost a full round of
# protenix-v2 and opendde folds here, and it reported a real opendde bug as absent.
#
# Two defences, both required:
#   1. cd to a NEUTRAL directory that contains no tt_bio/.
#   2. every arm resolves tt_bio.worker.__file__ first and ABORTS unless it is under the tree
#      that arm claims. An arm that cannot prove which code it ran is not evidence.
#
#   scripts/protenix_v1_port/ab_trees.sh <tree_a> <tree_b> <model> <target> [flags...]
set -euo pipefail

A=${1:?tree A}; B=${2:?tree B}; MODEL=${3:?model}; TARGET=${4:?target}; shift 4
FLAGS=("$@")
PY=${PY:-/home/moritz/tt-bio/env/bin/python3}
OUT=${OUT:-/tmp/ab_trees}
cd /tmp                                   # (1) neutral cwd

arm() {  # tag tree
  local tag=$1 tree=$2 got
  got=$($PY -c "import tt_bio.worker as W; print(W.__file__)" 2>/dev/null) \
    || { echo "ARM $tag: tt_bio did not import" >&2; exit 2; }
  case "$got" in "$tree"/*) ;; *)                                   # (2) prove the tree
    echo "ARM $tag CLAIMED $tree BUT RESOLVED $got -- refusing to record this as evidence" >&2
    exit 2;; esac
  echo "ARM $tag tree=$got" >&2
  rm -rf "$OUT/$tag"
  timeout 3000 $PY -m tt_bio.main predict "$TARGET" --model "$MODEL" "${FLAGS[@]}" \
    --out_dir "$OUT/$tag" >/dev/null 2>&1
  sha256sum "$(ls "$OUT/$tag"/*/structures/*.cif | head -1)" | cut -c1-16
}

mkdir -p "$OUT"
ha=$(PYTHONPATH=$A arm a "$A")
hb=$(PYTHONPATH=$B arm b "$B")
echo "a=$ha  b=$hb  ->  $([ "$ha" = "$hb" ] && echo IDENTICAL || echo DIFFERS)"
