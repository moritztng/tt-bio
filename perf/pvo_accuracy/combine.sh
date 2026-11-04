#!/bin/bash
# Put arms from different run roots side by side under new mode names, so analyze.py can compare any two of them
# (GPU exact vs TT exact, WH vs BH, abag vs general ckpt). No rescoring: the arm dirs are symlinked and the rows of
# each source preds.tsv are copied with the mode column renamed.
#   bash combine.sh <out root> <newmode>=<src root>:<src mode> ...
#   e.g. bash combine.sh gpu-vs-wh/opendde gpuexact=gpu/opendde:exact ttexact=tt/wh/opendde:ttexact
# Then: venv/bin/python analyze.py <data> <out root> <out root>/preds.tsv gpuexact ttexact > report.md
set -euo pipefail
out=$1; shift
mkdir -p "$out"
first=1
for spec in "$@"; do
  new=${spec%%=*} src=${spec#*=}; root=${src%:*} mode=${src##*:}
  for a in "$root"/"$mode"_*/; do a=${a%/}; ln -sfn "$(realpath "$a")" "$out/$new${a##*/$mode}"; done
  if [ $first = 1 ]; then head -1 "$root/preds.tsv" > "$out/preds.tsv"; first=0; fi
  awk -F'\t' -v OFS='\t' -v m="$mode" -v n="$new" 'NR > 1 && $2 == m { $2 = n; print }' "$root/preds.tsv" >> "$out/preds.tsv"
done
