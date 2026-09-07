# Walk the OpenBind-0 capacity ladder on ONE card, resume-safe.
#
#   sh perf/ceiling_openbind/run_ladder.sh <tree> <out-dir> <rung> [rung ...]
#
# The card is fixed by CARD (UMD index) and never picked: this box is the live JapanFold Galaxy
# and 26 of its 32 chips are serving users, so a picker that scans is a picker that can land on
# one of them. The /dev node is only for the occupancy check -- a UMD index is not a node number
# (ceilof3 lost a whole pass to that), so NODE is given explicitly and asserted against
# pick_chip.py's map before anything opens.
#
# `--override` always: predict applies resume semantics without it, so a re-run into a populated
# out_dir folds nothing, exits in seconds and logs the PREVIOUS verdict as this one's.
#
# The engine under test is logged as tt_bio.__file__, not as a branch name: a label can lie about
# which tree got imported and a module path cannot.
set -u
TREE=$1; OUT=$2; shift 2
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
CARD=${CARD:-1}
NODE=${NODE:-17}
LOG=$OUT/ladder.log
RUNGS=$(cd "$TREE/rundir" && pwd)/rungs
mkdir -p "$OUT"

map=$(cd "$TREE" && "$PY" perf/ceiling_of3/pick_chip.py --map 2>/dev/null)
case " $map " in
  *" $CARD->$NODE "*) : ;;
  *) echo "REFUSED card $CARD is not node $NODE in [$map]" >> "$LOG"; exit 2 ;;
esac

for r in "$@"; do
  if grep -q "^RUNG $r " "$LOG" 2>/dev/null; then
    echo "=== $r already folded, skipping" >> "$LOG"
    continue
  fi
  if sudo -n lsof "/dev/tenstorrent/$NODE" >/dev/null 2>&1; then
    echo "UNRUN $r node $NODE busy $(date -u +%FT%TZ)" >> "$LOG"
    sleep 60
    continue
  fi
  s=$(date +%s)
  engine=$(cd "$TREE" && PYTHONPATH="$TREE" "$PY" -c 'import tt_bio, sys; sys.stdout.write(tt_bio.__file__)')
  echo "=== $r start card=$CARD node=$NODE engine=$engine $(date -u +%FT%TZ)" >> "$LOG"
  ( cd "$TREE" && TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
      TT_BIO_LEASE_HOLDER=worker:ceiling-openbind-1024 TT_METAL_LOGGER_LEVEL=FATAL \
      PYTHONPATH="$TREE" "$PY" -m tt_bio.main predict "$RUNGS/$r.yaml" \
      --model openbind --accelerator tenstorrent --out_dir "$OUT/$r" --override \
      --msa_dir "$TREE/rundir/msacache_deep" --msa_cache_only --debug ) > "$OUT/$r.log" 2>&1
  rc=$?
  e=$(date +%s)
  if grep -q "DeviceInUseError" "$OUT/$r.log" 2>/dev/null; then
    echo "UNRUN $r lost card=$CARD to a lease holder $(date -u +%FT%TZ)" >> "$LOG"
    sleep 30
    continue
  fi
  st=$("$PY" - "$OUT/$r" <<'PYEOF2' 2>/dev/null || echo NORESULT
import glob, json, sys
g = glob.glob(sys.argv[1] + "/*/results.json")
print(json.load(open(g[0]))[0]["status"] if g else "NORESULT")
PYEOF2
)
  refused=$(grep -oE "Not enough space to allocate [0-9]+ B" "$OUT/$r.log" 2>/dev/null | head -1 | tr -d '\n')
  echo "RUNG $r rc=$rc status=$st wall=$((e - s))s card=$CARD ${refused:+refused=[$refused]} $(date -u +%FT%TZ)" >> "$LOG"
done
echo "LADDER DONE $(date -u +%FT%TZ)" >> "$LOG"
