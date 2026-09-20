#!/usr/bin/env bash
# The cross-model Angstrom reading TT_BIO_APB_CONCAT_HEADS still owes, on any model that runs it.
#
# Generalises apb_pvx_accuracy.sh, which did this for protenix-v2 only. Same five-fold design,
# model/target/steps now arguments, because the gate's coverage column changed which model is the
# right one to spend folds on.
#
# WHY A GREEN GATE IS NOT AN ACCURACY READING. scripts/release_gate.py sets every model's floor at
# roughly 2x its measured value and its own comments say why -- "catches a gross failure, not
# run-to-run MSA-draw noise". The TT_BIO_UNFUSED_SILU regression Moritz refused in September was
# ~1.1 A of CA-RMSD on Protenix-v2 against 2.13 A of slack, so it would have gone green. A shared-
# code default flip needs a paired reading, not a floor.
#
# WHICH MODEL, AND WHY IT CHANGED. The flag is only evidence where it FIRES, and the gate now
# counts that per arm (perf/c14_land/gate_apb_summary.py):
#
#     boltz2        5096 served / 0 declined     full coverage, already scored at 0.2244 A
#     opendde       5304 served / 0 declined     full coverage, NEVER scored
#     protenix-v2    500 served / 4800 declined  ~9 % coverage, scored at 1.997 vs 1.999 A
#     protenix-v1    212 served / 4800 declined  ~4 % coverage, never scored
#     esmfold2         0 served                  no AttentionPairBias site at all
#
# So opendde is the strongest remaining reading: it is the only model besides Boltz-2 where every
# head re-assembly takes the new path, and protenix-v2's existing number covers less than it looks
# like -- 91 % of that arm is unchanged code.
#
# THE FIVE FOLDS. Paired on seed, flag the only difference, with the controls that make the signal
# readable at all:
#     off seed 0 run a   reference
#     on  seed 0 run a   signal
#     off seed 0 run b   A/A control -- must be 0.000000 A or nothing else here is readable
#     off seed 1, off seed 2   the seed-scatter floor the signal is judged against
#
# Angstrom, not seconds. No timing guard is taken and none is needed, so this is admissible on a
# loud host -- but NOT while this row's own gate is folding on the same card. Pass 25 cost 15 gate
# arms to exactly that.
#
#   MODEL=opendde TARGET=examples/prot.yaml STEPS=200 CARD=1 bash perf/c14_land/apb_xmodel_accuracy.sh
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail
PY=/home/ttuser/tt-bio-dev/env/bin/python3
MODEL=${MODEL:-opendde}
TARGET=${TARGET:-examples/prot.yaml}
STEPS=${STEPS:-200}
SAMPLES=${SAMPLES:-1}
CARD=${CARD:-1}
OUT=$WT/perf/c14_land/apb_${MODEL}_cifs
cd "$WT" || exit 1
mkdir -p "$OUT"

echo "=== APB cross-model accuracy: $MODEL on $TARGET, $STEPS steps, card $CARD, $(date -Is) ==="
echo "=== tree $(git rev-parse --short HEAD) ==="

one() {  # arm seed runlabel
  local arm=$1 seed=$2 run=$3
  local work=$WT/perf/c14_land/xm_scratch_${MODEL}_${arm}_${seed}_${run}
  rm -rf "$work"; mkdir -p "$work"
  env TT_BIO_APB_CONCAT_HEADS=$([ "$arm" = on ] && echo 1 || echo 0) \
      TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
      TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
      C14_GI_COUNTER=tt_bio.tenstorrent:APB_CONCAT_HEADS_STATS \
      C14_GI_OUT="$OUT/firing_${arm}_${seed}_${run}" \
      PYTHONPATH="$WT:$WT/perf/c14_land/gishim" \
      "$PY" -m tt_bio.main predict "$TARGET" --model "$MODEL" \
        --single_sequence --sampling_steps "$STEPS" --diffusion_samples "$SAMPLES" \
        --seed "$seed" --out_dir "$work" \
        > "$WT/perf/c14_land/xm_${MODEL}_${arm}_${seed}_${run}.log" 2>&1
  local rc=$?
  # Name the arm dir <size>_<arm>_<run>, which is what perf/other512/cif_rmsd.py parses.
  # The first run used <arm><seed>_<run> and that scorers per-arm rollup found no members and
  # crashed after printing the pairwise table -- the measurement survived, the summary did not.
  local d=$OUT/0_${arm}${seed}_${run}
  mkdir -p "$d"
  find "$work" -name '*.cif' -exec cp {} "$d"/ \;
  # The firing count travels with the fold. An arm that scored 0.000 A because the flag never
  # reached the site is not a pass, and without this it looks exactly like one.
  echo "$(date -u +%H:%M:%SZ) model=$MODEL arm=$arm seed=$seed run=$run rc=$rc cifs=$(ls "$d" | wc -l)"
  rm -rf "$work"
}

one off 0 a
one on  0 a
one off 0 b
one off 1 a
one off 2 a
echo "=== folds done $(date -Is); score with perf/c14_land/cif_rmsd-style scoring over $OUT ==="
echo "=== firing per arm ==="
for f in "$OUT"/firing_*.*; do
  python3 - "$f" <<'PYEOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit
if r.get("counter") and any(r["counter"]):
    print(" ", sys.argv[1].split("/")[-1], r["counter"])
PYEOF
done
