#!/bin/bash
# release_gate.py arms for the shipped models whose code the R4T merge touches (rfd3, pxdesign and
# protenix-v2 import dit_pair_bias / outer_residual, whose gates gained a tape decline), on the landing
# sha, one qb2 chip, booked normally. Each arm's verdict is its own exit code; the chain ends ALLDONE OK
# only if every arm exits 0.
#   bash r4t_land_rg.sh <chip> <outdir> <land_sha> [arm ...]
set -u
N=${1:?chip}; R=${2:?outdir}; LAND=${3:?land sha}; shift 3
ARMS=${*:-rfd3 pxdesign protenix-v2}
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
T=${T:-$HOME/r4t/land/trees}
BOOK="python3 $HOME/spd/feed/book.py"; OWN=fleet-r4t-land; SECS=${SECS:-10800}
mkdir -p "$R"; R=$(cd "$R" && pwd)
stamp(){ echo "$1 $(date -u +%FT%TZ)" | tee -a "$R/done"; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:r4t-land
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N $SECS --note 'r4t-land release arms' >/dev/null; sleep 60; done
stamp "window $N open"
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
git -C "$T" checkout -q --detach "$LAND" || { stamp "checkout FAIL"; exit 2; }
fails=0
for arm in $ARMS; do
  (cd "$T" && PYTHONPATH=$T timeout 3600 "$PY" scripts/release_gate.py --model "$arm") > "$R/$arm.log" 2>&1
  rc=$?; stamp "$arm rc=$rc sha=${LAND:0:9}"; [ $rc -eq 0 ] || fails=$((fails+1))
done
$BOOK rm $OWN >/dev/null 2>&1
[ $fails -eq 0 ] && stamp "ALLDONE OK" || stamp "ALLDONE FAIL ($fails)"
