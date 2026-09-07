# Run the standing release gate on ONE card for the models named, serially, resume-safe.
#
#   CARD=<umd> NODE=<dev-node> PY=<interpreter> sh perf/ceil1024_int/gate_chain.sh <tree> <out> <model>...
#
# Serial per card and never parallel within a card: one device context per process.
#
# The models are not a choice. OuterProductMean, PairWeightedAveraging and the MSA block are
# shared by boltz2, protenix-v2, opendde, openfold3, rf3 and openbind, and the merge changed all
# three. "openfold3 still folds" is not the question -- boltz2 is, because it folds 1024 in
# production today and is the one a wrong resolution would break first.
#
# TT_BIO_CAPACITY_CENSUS is on for every leg. The merge's whole claim is that a fold which is
# never refused keeps its exact path, and that claim is an argument until dram_narrowed and
# join_split are read back as 0. The counters cost no arithmetic; only the atexit dump is new.
set -u
TREE=$1; OUT=$2; shift 2
PY=${PY:-/home/cust-team/mthuening/gate-env/bin/python3.10}
CARD=${CARD:?set CARD to the UMD index}
NODE=${NODE:?set NODE to the /dev/tenstorrent node for CARD}
mkdir -p "$OUT"
LOG=$OUT/gate.log

# A UMD index is not a /dev node number on this box. Assert the pairing before anything opens,
# or the freeness check reads a chip the gate never touches.
map=$(cd "$TREE" && "$PY" perf/ceiling_of3/pick_chip.py --map 2>/dev/null)
case " $map " in
  *" $CARD->$NODE "*) : ;;
  *) echo "REFUSED card $CARD is not node $NODE in [$map]" >> "$LOG"; exit 2 ;;
esac

for m in "$@"; do
  grep -q "^GATE $m " "$LOG" 2>/dev/null && continue
  waited=0
  while sudo -n lsof "/dev/tenstorrent/$NODE" >/dev/null 2>&1; do
    waited=$((waited + 1))
    if [ "$waited" -gt 240 ]; then
      echo "UNRUN $m node $NODE busy for 2h $(date -u +%FT%TZ)" >> "$LOG"
      break
    fi
    sleep 30
  done
  s=$(date +%s)
  ( cd "$TREE" && TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:ceiling-1024-integration-and-gate \
      TT_BIO_CAPACITY_CENSUS="$OUT/census_$m" PYTHONPATH="$TREE" \
      "$PY" scripts/release_gate.py --model "$m" ) > "$OUT/$m.log" 2>&1
  rc=$?
  echo "GATE $m rc=$rc wall=$(($(date +%s) - s))s card=$CARD sha=$(git -C "$TREE" rev-parse --short HEAD) $(date -u +%FT%TZ)" >> "$LOG"
done
echo "GATE CHAIN DONE card=$CARD $(date -u +%FT%TZ)" >> "$LOG"
