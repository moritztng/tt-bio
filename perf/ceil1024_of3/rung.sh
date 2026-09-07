#!/bin/sh
# One OpenFold3 deep-MSA rung on the ONE card this workstream was lent (UMD 0 of the
# production Galaxy). Everything else on that box is serving users, so the card is
# never chosen and never widened: it is an argument, and a rung that cannot have it
# runs nothing rather than reaching for a sibling.
#
#   TREE=/path/to/checkout OUT=rundir/out_x CARD=0 sh rung.sh tile_1024 ...
#
# Resume-safe on `RUNG ` lines: a relaunch skips only a size that produced a result, so a
# lost device race or a killed turn never leaves a hole recorded as a measurement.
set -u
RUN=${RUN:-/home/cust-team/mthuening/ceil1024/rundir}
TREE=${TREE:-/home/cust-team/mthuening/ceil1024of3}
OUT=${OUT:-$RUN/out}
CARD=${CARD:-0}
MODEL=${MODEL:-openfold3}
FIX=${FIX:-msafix_tile}
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
LOG=$OUT/ladder.log
cd "$RUN" || exit 1
mkdir -p "$OUT"

# Which tt_bio actually gets imported. A label can lie about the arm; a module path cannot.
engine=$(PYTHONPATH="$TREE" "$PY" -c 'import tt_bio, os; print(os.path.dirname(tt_bio.__file__))')
case "$engine" in
  "$TREE"/*) ;;
  *) echo "REFUSING: TREE=$TREE but tt_bio resolves to $engine" >> "$LOG"; exit 4 ;;
esac
echo "=== ladder start tree=$TREE@$(git -C "$TREE" rev-parse --short HEAD) engine=$engine out=$OUT card=$CARD $(date -u +%FT%TZ)" >> "$LOG"

for r in "$@"; do
  if grep -q "^RUNG $r " "$LOG" 2>/dev/null; then
    echo "=== $r already folded, skipping" >> "$LOG"; continue
  fi
  s=$(date +%s)
  echo "=== $r start card=$CARD $(date -u +%FT%TZ)" >> "$LOG"
  # --override, always: without it `predict` resumes and reports the PREVIOUS verdict as this
  # run's. TT_BIO_SIZE_LIMIT=0: size_limits refuses above the last measured cap, and a ladder
  # whose job is to find the next cap cannot be bounded by the previous one.
  TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
    TT_BIO_LEASE_HOLDER=worker:ceiling-openfold3-1024 TT_BIO_LEASE_TIMEOUT=20 \
    TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$TREE" TT_BIO_SIZE_LIMIT=0 ${EXTRA_ENV:-} \
    "$PY" -m tt_bio.main predict "$FIX/$r.yaml" \
    --model "$MODEL" --accelerator tenstorrent --out_dir "$OUT/$r" --override \
    --msa_dir msacache_deep --msa_cache_only --debug > "$OUT/$r.log" 2>&1
  rc=$?
  e=$(date +%s)
  if grep -q "DeviceInUseError" "$OUT/$r.log" 2>/dev/null; then
    echo "UNRUN $r lost card=$CARD to a lease holder $(date -u +%FT%TZ)" >> "$LOG"; continue
  fi
  st=$("$PY" - "$OUT/$r" <<'PYEOF' 2>/dev/null || echo NORESULT
import glob, json, sys
g = glob.glob(sys.argv[1] + "/*/results.json")
print(json.load(open(g[0]))[0]["status"] if g else "NORESULT")
PYEOF
)
  echo "RUNG $r rc=$rc status=$st wall=$((e - s))s card=$CARD $(date -u +%FT%TZ)" >> "$LOG"
done
echo "LADDER DONE $(date -u +%FT%TZ)" >> "$LOG"
