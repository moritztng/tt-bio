#!/bin/bash
# DockQ guard for the kit arms, on pc. Maps the box layout <arm>/<PDB>/r<k>/p0/<PDB>/seed_<s>/ onto pvo-accuracy's
# score.py layout <arm>/pred/<PDB>/seed_<s>/ with symlinks, then scores every sample against the deposited structure.
#   score_guard.sh <cif root: holds opendde/<arm>/... and mol1/<arm>/...> <out dir>
set -eu
SRC=$(realpath "$1"); OUT=$2; D=$HOME/pvo-accuracy-data
SCORE=${SCORE:-$HOME/.coworker/wt/pvo-kit/perf/pvo_kit/score.py}
mkdir -p "$OUT"
for m in "$SRC"/*/; do m=$(basename "$m")
  T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
  for s in "$SRC"/$m/*/*/r*/p0/*/seed_*; do
    a=$(echo "$s" | sed "s|$SRC/$m/||" | cut -d/ -f1); p=$(basename "$(dirname "$s")")
    [ "$a" = gate ] && continue
    n=${a//-/}_0   # score.py reads the mode as \w+ before _<digits>: kit-fast -> kitfast
    mkdir -p "$T/$n/pred/$p"; ln -sfn "$s" "$T/$n/pred/$p/$(basename "$s")"
  done
  $D/venv/bin/python "$SCORE" "$D" "$T" > "$OUT/$m.tsv" 2> "$OUT/$m.err"
  rm -rf "$T"; trap - EXIT
done
