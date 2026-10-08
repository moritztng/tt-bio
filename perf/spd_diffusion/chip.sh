#!/bin/bash
# spd-diffusion on one chip, everything under the chip's flock: the conditioning (once per run dir),
# then run.sh's arms, then the traced arms. Engines and data under $B (eng/<name>, spd-data).
# usage: chip.sh CHIP LOCK RUN "arm=ENG[:ENV=V,...]"... [-- "traced arm"...]
# env: B (default ~/spd/spd-diffusion), PY (python), WAIT (flock -w seconds, default 21600)
CHIP=$1; LOCK=$2; R=$3; shift 3
B=${B:-$HOME/spd/spd-diffusion}; PY=${PY:-python}; WAIT=${WAIT:-21600}
export SPD_DATA=${SPD_DATA:-$B/spd-data} TT_BIO_LEASE_DIR=$B/leases TT_BIO_LEASE_HOLDER=spd-diffusion
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$R"
plain=(); traced=(); t=0
for a in "$@"; do [ "$a" = -- ] && { t=1; continue; }; ((t)) && traced+=("$a") || plain+=("$a"); done
mkdir -p "$B/bin"; ln -sf "$(command -v $PY)" "$B/bin/python"; export PATH=$B/bin:$PATH
exec 9>"$LOCK"; flock -w "$WAIT" 9 || { echo "$(date -u +%FT%TZ) no lock" >> "$R/driver.log"; exit 1; }
echo "$(date -u +%FT%TZ) lock held, chip $CHIP" >> "$R/driver.log"
if [ ! -s "$R/cond.pt" ]; then
  TT_VISIBLE_DEVICES=$CHIP PYTHONPATH=$B/eng/diff timeout 2400 python "$HERE/bench.py" cond "$R" "$CHIP" \
    "$SPD_DATA/inputs/c730.yaml" > "$R/cond.log" 2>&1
  echo "$(date -u +%FT%TZ) cond rc=$?" >> "$R/driver.log"
fi
((${#plain[@]})) && "$HERE/run.sh" "$R/arms" "$CHIP" "$R/cond.pt" "${plain[@]}"
((${#traced[@]})) && SPD_TRACE=1 "$HERE/run.sh" "$R/traced" "$CHIP" "$R/cond.pt" "${traced[@]}"
echo "$(date -u +%FT%TZ) end" >> "$R/driver.log"
