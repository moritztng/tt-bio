# Walk the OpenBind-0 capacity ladder on ONE card, resume-safe.
#
#   sh perf/ceiling_openbind/run_ladder.sh <tree> <out-dir> <rung> [rung ...]
#
# MODEL, RUNGS and MSA override the defaults. OpenBind-0 and OpenFold3 are the same OF3Trunk on
# two checkpoints, so a shared-trunk capacity fix has to be walked on BOTH before it is a
# shared fix rather than a claim -- same runner, same card, different MODEL and fixtures.
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
# The engine under test is logged as tt_bio.__file__ AND as the tree's HEAD sha. The path proves
# which tree got imported, which a branch name cannot; the sha proves which COMMIT, which the path
# cannot. Both are needed, and the second was learned the hard way: a `git checkout` in the tree
# mid-walk (to re-test a harness script) silently moved the engine between rungs, and every line
# in the log said the same thing because the path never changed. It was harness-only that time,
# so the ladder survived; it would not have been visible if it had not been.
set -u
TREE=$1; OUT=$2; shift 2
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
CARD=${CARD:-1}
NODE=${NODE:-17}
MODEL=${MODEL:-openbind}
LOG=$OUT/ladder.log
RUNGS=${RUNGS:-$(cd "$TREE/rundir" && pwd)/rungs}
MSA=${MSA:-$TREE/rundir/msacache_deep}
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
  attempt=0
  while [ "$attempt" -lt 4 ]; do
    attempt=$((attempt + 1))
    # Waiting for the card is not an attempt. Four attempts of sixty seconds is a four-minute
    # patience, and one rung of this ladder takes five, so a chained arm would have given up
    # before the arm ahead of it finished a single fold.
    waited=0
    while sudo -n lsof "/dev/tenstorrent/$NODE" >/dev/null 2>&1; do
      waited=$((waited + 1))
      if [ "$waited" -gt 240 ]; then
        echo "UNRUN $r node $NODE busy for 2h $(date -u +%FT%TZ)" >> "$LOG"
        break
      fi
      sleep 30
    done
    if [ "$waited" -gt 240 ]; then continue; fi
    s=$(date +%s)
    engine=$(cd "$TREE" && PYTHONPATH="$TREE" "$PY" -c 'import tt_bio, sys; sys.stdout.write(tt_bio.__file__)')
    sha=$(git -C "$TREE" rev-parse --short HEAD 2>/dev/null)
    echo "=== $r attempt $attempt start card=$CARD node=$NODE sha=$sha engine=$engine $(date -u +%FT%TZ)" >> "$LOG"
    # TT_BIO_SIZE_LIMIT=0: the ceiling under test is exactly what size_limits refuses on, so a
    # ladder that honoured it could only ever re-measure the published number.
    # The counters are what turn "this rung took a byte-identical path" from an argument into a
    # reading: dram_narrowed and join_split at 0 means no capacity fallback ran.
    ( cd "$TREE" && TT_BIO_SIZE_LIMIT=0 TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
        TT_BIO_CAPACITY_CENSUS="$OUT/$r.census" \
        TT_BIO_LEASE_HOLDER=worker:ceiling-openbind-1024 TT_METAL_LOGGER_LEVEL=FATAL \
        PYTHONPATH="$TREE" "$PY" -m tt_bio.main predict "$RUNGS/$r.yaml" \
        --model "$MODEL" --accelerator tenstorrent --out_dir "$OUT/$r" --override \
        --msa_dir "$MSA" --msa_cache_only --debug ) > "$OUT/$r.log" 2>&1
    rc=$?
    e=$(date +%s)
    if grep -q "DeviceInUseError" "$OUT/$r.log" 2>/dev/null; then
      echo "UNRUN $r attempt $attempt lost card=$CARD to a lease holder $(date -u +%FT%TZ)" >> "$LOG"
      sleep 30
      continue
    fi
    # A chip left dirty by the previous process fails INSIDE the device open ("Read unexpected
    # run_mailbox value"), before a single op runs. That is not a rung: recording it as one makes
    # the resume check skip a size that was never folded, which is how a ceiling gets published
    # with a hole in its ladder. Observed on this card at 2026-09-07T10:36Z, and the next open
    # came up clean, so retrying the rung is the whole recovery.
    if grep -q "run_mailbox" "$OUT/$r.log" 2>/dev/null; then
      echo "UNRUN $r attempt $attempt card=$CARD came up dirty (run_mailbox) $(date -u +%FT%TZ)" >> "$LOG"
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
    cens=$("$PY" - "$OUT/$r.census" <<'PYEOF3' 2>/dev/null
import glob, json, sys
# Every process that imported tt_bio writes one file: the parent's counters are all zero and the
# spawned worker's carry the fold. Summing takes the worker without having to guess which pid it is.
g = glob.glob(sys.argv[1] + "/capacity_*.json")
if g:
    d = [json.load(open(f)) for f in g]
    t = lambda k, s: sum(x[k][s] for x in d)
    print("narrowed=%d/%d/%d join_split=%d" % (
        t("opm_row", "dram_narrowed"), t("pwa_depth", "dram_narrowed"),
        t("fp32_softmax", "dram_narrowed"), t("opm_row", "join_split")))
PYEOF3
)
    echo "RUNG $r rc=$rc status=$st wall=$((e - s))s card=$CARD ${refused:+refused=[$refused]} ${cens:+$cens} $(date -u +%FT%TZ)" >> "$LOG"
    break
  done
done
echo "LADDER DONE $(date -u +%FT%TZ)" >> "$LOG"
