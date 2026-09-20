#!/usr/bin/env bash
# Release gate for TT_BIO_APB_CONCAT_HEADS=1, every arm except the wall-clock one.
#
# WHY THIS CAN RUN ON A BUSY BOX. The gate's arms are correctness and accuracy reads -- CIF
# digests, RMSD and TM against reference structures, clash counts, capacity. None of those move
# with the host's loadavg; they only take longer. `size-ladder` is the exception: it is the gate's
# one TIMED arm, it takes no benchlock and runs no guard, and it has already produced two false
# reds in this campaign on a quiet box. It is EXCLUDED here and owed separately on a quiet host,
# not skipped silently.
#
# ARM NAMES COME FROM `--list-arms`, never hand-copied. An earlier gate in this row lost its
# OpenBind arm to `--model openbind-0`, the model's prose name, which argparse refused in the same
# second it started; eleven green arms plus one never attempted reads as twelve.
#
# TWO THINGS LEARNED THE HARD WAY ON 2026-09-20, both fixed here:
#
# 1. STOP ON A PREFLIGHT REFUSAL, do not spend the arm list on it. The gate refuses to start when
#    1-min loadavg is over 1.5x nproc. On the first run that burned all 21 arms in five seconds;
#    on the second it burned the last 15 in three, because I had launched a four-fold job of my own
#    on another card and became a co-tenant of my own gate. A refusal is "come back later", not a
#    result, so the loop now waits and RETRIES THE SAME ARM instead of advancing past it. Without
#    that, a run reports 15 reds that are not reds -- `gatechain-no-failure-stop-relaunches-next-
#    arm-on-dead-card` in a new costume.
#
# 2. ITS OWN JOURNAL FILE. `gate_journal.KEY_FIELDS` is
#    (commit, dirty, host, card_type, fast, diffusion_trace, package) and carries NO env-flag
#    component, while its own docstring calls itself "everything that changes what an arm would
#    score". A TT_BIO_* flag plainly changes what an arm scores. The commit field saves the usual
#    case, but two runs of DIFFERENT flags at the SAME commit -- which is exactly how a flag gets
#    gated -- would cross-credit each other's arms, and `l1-budget` is the arm that decides the
#    grid hard stop. A per-flag journal cannot collide with anything.
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=1
LEDGER="$WT/perf/c14_land/gate_apb_ledger.txt"
JOURNAL="$WT/perf/c14_land/gate_apb_journal.jsonl"
MAX_WAIT_S=${MAX_WAIT_S:-5400}      # per arm, total time willing to wait out a loud box
cd "$WT" || exit 1
export PYTHONPATH="$WT"           # score THIS tree, not the shared checkout

ARMS=$("$PY" scripts/release_gate.py --list-arms | grep -vx 'size-ladder')
echo "=== APB GATE START $(date -Is) commit $(git rev-parse --short HEAD) card $CARD ==="
echo "arms: $(echo $ARMS | tr '\n' ' ')"
echo "excluded: size-ladder (timed arm)"
echo "journal: $JOURNAL (per-flag, so it cannot inherit another flag's arms)"
: > "$LEDGER"
for ARM in $ARMS; do
  WAITED=0
  while : ; do
    echo
    echo "##### ARM $ARM START $(date -Is) load $(cut -d' ' -f1 /proc/loadavg) #####"
    LOG="perf/c14_land/gate_apb_${ARM}.log"
    TT_BIO_APB_CONCAT_HEADS=1 TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
      "$PY" scripts/release_gate.py --keep --journal "$JOURNAL" --model "$ARM" 2>&1 \
      | tee "$LOG"
    RC=${PIPESTATUS[0]}
    # A preflight refusal is not a verdict. Distinguish it from a real red by the gate's own
    # words, wait, and re-run THIS arm rather than moving on.
    if [ "$RC" -ne 0 ] && grep -q 'PREFLIGHT - refusing to run the gate' "$LOG"; then
      if [ "$WAITED" -ge "$MAX_WAIT_S" ]; then
        printf '%s\t%s\tREFUSED-GAVE-UP after %ss\n' "$(date -Is)" "$ARM" "$WAITED" >> "$LEDGER"
        echo "##### ARM $ARM REFUSED, waited ${WAITED}s, giving up $(date -Is) #####"
        break
      fi
      echo "##### ARM $ARM PREFLIGHT REFUSED, waiting 300s (waited ${WAITED}s) #####"
      sleep 300; WAITED=$((WAITED + 300)); continue
    fi
    echo "##### ARM $ARM END rc=$RC $(date -Is) #####"
    printf '%s\t%s\trc=%s\n' "$(date -Is)" "$ARM" "$RC" >> "$LEDGER"
    break
  done
done
echo "=== APB GATE END $(date -Is) ==="
