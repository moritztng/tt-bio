#!/bin/bash
# harvest_par.py pulls from <run>/<mdir>/<target>_c<chunk>/<mdir>_results_<target>/.
# od9j4c_fleet.sh writes to <run>/<target>_c<chunk>/ and chunk 0 lives in p34d/odcamp,
# so give harvest the layout it expects with symlinks. Directories, not copies: the
# folds are live and these resolve as they fill.
set -u
H=$HOME/mthuening
B=$H/p34d/od9j4c
mkdir -p "$B/opendde"
for c in 1 2 3 4 5 6 7; do
  ln -sfn "../9j4c_c$c" "$B/opendde/9j4c_c$c"
done
ln -sfn "../../odcamp/9j4c_c0" "$B/opendde/9j4c_c0"
ls -l "$B/opendde" | sed 's/^/  /'
