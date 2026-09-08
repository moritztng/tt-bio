#!/bin/bash
# The 512 rung on two engines: same chip, same fixture, same flags, ONE variable.
#
# usage: px_512ab.sh <chip> <this-checkout> <other-checkout> [seed]
#
# The variable is PYTHONPATH, not a second venv. An `__editable__` finder installs
# itself with sys.meta_path.append, i.e. AFTER the stdlib PathFinder, so PYTHONPATH
# outranks an editable install and one venv can serve both arms with an identical
# dependency set. That is not a shortcut, it is the correction: the earlier version of
# this script ran its old-engine arm out of a checkout whose `env` was a SYMLINK to
# another venv, so both arms imported the same tt_bio and would have agreed no matter
# what. Hence the assertion below -- an A/B that cannot fail loudly is worse than none.
#
# Seed matters as much as the engine here. 512 is the one rung whose fit residual is an
# order of magnitude above the rest (2.4-3.0 A against 0.13-0.20 A), which is the regime
# where the seed moves the answer more than 300 commits of engine drift do: seed 0 gives
# 2.4378 and seed 42 gives 2.9924 on the SAME engine. Compare against a recorded number
# only at the seed it was recorded on.
set -u
chip=$1; a=$2; b=$3; seed=${4:-0}
export TT_METAL_LOGGER_LEVEL=FATAL
export TT_VISIBLE_DEVICES=$chip TT_BIO_LEASE_CARDS=$chip
export TT_BIO_LEASE_TIMEOUT=${TT_BIO_LEASE_TIMEOUT:-900}
py=${PX_PYTHON:-$a/env/bin/python}
out=${PX_OUT:-/tmp/px512ab.$$}

for eng in "$a" "$b"; do
  o=$out/$(basename "$eng"); rm -rf "$o"; mkdir -p "$o"
  got=$(PYTHONPATH=$eng $py -c 'import tt_bio; print(tt_bio.__file__)' 2>&1)
  if [ "$got" != "$eng/tt_bio/__init__.py" ]; then
    echo "RESULT engine=$eng ENGINE_MISMATCH got=$got"; continue
  fi
  t0=$(date +%s.%N)
  PYTHONPATH=$eng timeout 1800 $py -u -m tt_bio.main design \
    "$eng/perf/pxdesign/targets/ladder_512.yaml" --model pxdesign \
    --num_designs 1 --n_step 400 --seed "$seed" --out_dir "$o/designs" > "$o/run.log" 2>&1
  rc=$?; t1=$(date +%s.%N)
  fit=$(grep -o '"fit_rmsd": [0-9.]*' "$o"/designs/*.json 2>/dev/null | sed 's/.*: //' | tr '\n' ',')
  [ -n "$fit" ] || fit=$(grep -o 'target fit [0-9.]* A' "$o/run.log" | tail -1)
  printf 'RESULT engine=%s chip=%s seed=%s rc=%s wall_s=%.1f fit=%s\n' \
    "$(basename "$eng")" "$chip" "$seed" "$rc" "$(echo "$t1-$t0"|bc)" "${fit:-none}"
done
