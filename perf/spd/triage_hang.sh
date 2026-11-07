#!/bin/bash
# Run by tt-metal on a dispatch timeout (hangwatch.sh DIAG=1): tt-triage against the stuck chip, before the host
# throws. Needs tt-metal's tools/triage at the wheel's version and a venv with its requirements (TRIAGE_HOME,
# TRIAGE_PY). The fold's environment is inherited, so TT_VISIBLE_DEVICES scopes it to the one chip.
TRIAGE_HOME=${TRIAGE_HOME:-$HOME/spd/spd-fasthang/tm}
TRIAGE_PY=${TRIAGE_PY:-$HOME/spd/spd-fasthang/xv/bin/python}
out=${FH_OUT:-/tmp}/triage.txt
{
  echo "$(date -u +%FT%TZ) dispatch timeout, chip ${TT_VISIBLE_DEVICES:-?}"
  cd "$TRIAGE_HOME" && timeout 900 "$TRIAGE_PY" tools/tt-triage.py --disable-progress --disable-colors -vv
  echo "rc=$?"
} >> "$out" 2>&1
