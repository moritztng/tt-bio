#!/bin/bash
# INFERENCE-AB for r4t-land: RFD4 design, grad off, on one qb2 chip.
#   A/A floor  = the SAME sha (main) run twice on the same chip.
#   A/B        = the step branch and the merged landing tree against that floor.
#   abb3       = scripts/abb3_port model_gate + step_gate on the landing tree (re-records the two
#                RECORDED-AT artifacts whose import closure the merge moved).
# Shas are pinned arguments, pushed into $T from pc beforehand (qb2 has no GitHub access), so a stale
# remote ref cannot silently substitute an older tree. The driver (perf/rfd4/e2e/design.py and the perf/
# modules it imports) is taken from the landing sha for every leg, so all legs run one harness and only
# the model code differs.
# A leg is OK only if design.py exits 0 AND wrote exactly DESIGNS CIFs; otherwise it is stamped FAIL and
# the chain ends ALLDONE FAIL.
#   bash r4t_land_ab.sh <chip> <outdir> <main_sha> <step_sha> <land_sha>
#   R4T_AB_DRY=1 PY=<python> ...   skip booking/flock/chip wait (used to show a leg can FAIL)
set -u
N=${1:?chip}; R=${2:?outdir}; MAIN=${3:?main sha}; STEP=${4:?step sha}; LAND=${5:?land sha}
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
ABB3_PY=${ABB3_PY:-$HOME/tt-bio-dev/env/bin/python}
T=${T:-$HOME/r4t/land/trees}; IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; DESIGNS=2
BOOK="python3 $HOME/spd/feed/book.py"; OWN=fleet-r4t-land
mkdir -p "$R"; R=$(cd "$R" && pwd); D=$R/driver
stamp(){ echo "$1 $(date -u +%FT%TZ)" | tee -a "$R/done"; }
fails=0
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:r4t-land

if [ -z "${R4T_AB_DRY:-}" ]; then
  # No --now: wait for our own window instead of overlaying a hold on another row's chip.
  until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N 7200 --note 'r4t-land INFERENCE-AB' >/dev/null; sleep 60; done
  stamp "window $N open"
  exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
  clear=0
  for _ in $(seq 360); do
    if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q "/dev/tenstorrent/$N\$"; then clear=0; else clear=$((clear+1)); fi
    [ $clear -ge 3 ] && break; sleep 10
  done
  [ $clear -ge 3 ] || { stamp "chip $N never free"; exit 1; }
fi

rm -rf "$D"; mkdir -p "$D"
git -C "$T" archive "$LAND" perf | tar -x -C "$D" || { stamp "driver export FAIL"; exit 2; }

run_leg(){ # leg sha
  local leg=$1 sha=$2 rc n
  git -C "$T" checkout -q --detach "$sha" || { stamp "$leg checkout FAIL"; fails=$((fails+1)); return; }
  rm -rf "${R:?}/$leg"
  (cd "$T" && PYTHONPATH=$T:$D/perf timeout 3600 "$PY" "$D/perf/rfd4/e2e/design.py" --spec "$IN/pdl1-128.yaml" \
     --weights "$W" --out "$R/$leg" --seed 0 --designs $DESIGNS --rng upstream) > "$R/$leg.log" 2>&1
  rc=$?
  n=$(ls "$R/$leg"/*_generated.cif 2>/dev/null | wc -l)
  if [ $rc -eq 0 ] && [ "$n" -eq $DESIGNS ]; then
    stamp "$leg OK sha=${sha:0:9} rc=$rc cifs=$n"
    sha256sum "$R/$leg"/*_generated.cif >> "$R/digests.txt"
  else
    stamp "$leg FAIL sha=${sha:0:9} rc=$rc cifs=$n"; fails=$((fails+1))
  fi
  grep -o '"aiclk[^,]*' "$R/$leg.log" | tail -4 | sed "s/^/$leg /" >> "$R/aiclk.txt"
}
run_leg aa1 "$MAIN"
run_leg aa2 "$MAIN"
run_leg ab_step "$STEP"
run_leg ab_land "$LAND"
for leg in aa2 ab_step ab_land; do
  for a in "$R/aa1"/*_generated.cif; do
    b="$R/$leg/$(basename "$a")"
    [ -f "$a" ] && [ -f "$b" ] && PYTHONPATH=$T "$PY" "$D/perf/rfd4_land/cif_compare.py" "$b" "$a" \
        --json "$R/cmp-$leg-$(basename "$a" .cif).json" >> "$R/cmp.txt" 2>&1
  done
done

if [ -z "${R4T_AB_DRY:-}" ]; then
  git -C "$T" checkout -q --detach "$LAND"
  mkdir -p "$R/abb3"
  for sigma in 0.05 0.3; do for tokens in 64 256; do
    (cd "$T" && PYTHONPATH=$T timeout 1800 "$ABB3_PY" scripts/abb3_port/model_gate.py --tokens $tokens --sigma $sigma)
  done; done > "$R/abb3/model_gate.txt" 2>&1
  stamp "abb3 model_gate rc=$? (last arm; grade by its rows)"
  (cd "$T" && PYTHONPATH=$T timeout 3600 "$ABB3_PY" scripts/abb3_port/step_gate.py) > "$R/abb3/step_gate.txt" 2>&1
  stamp "abb3 step_gate rc=$?"
  (cd "$T" && PYTHONPATH=$T timeout 1800 "$ABB3_PY" scripts/abb3_port/step_gate.py --micro 8) > "$R/abb3/step_gate_micro8.txt" 2>&1
  stamp "abb3 step_gate --micro 8 rc=$? (expected to OOM)"
  exec 9>&-; $BOOK rm $OWN
fi
[ $fails -eq 0 ] && stamp "ALLDONE OK" || { stamp "ALLDONE FAIL fails=$fails"; exit 1; }
