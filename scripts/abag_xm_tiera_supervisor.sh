#!/usr/bin/env bash
# abag_xm_tiera_supervisor.sh -- keep Tier-A generation alive on this host.
#
# Why this replaces abag_xm_tiera_watchdog.sh: the campaign lost 7.5 h on 2026-07-28 because all
# four drivers died at ~00:16 and nothing noticed until morning. The old watchdog was not a safety
# net -- it had two bugs that made it worse than nothing:
#   1. It killed folds with `pgrep -f "$WT.*tt_bio.main predict"`, a pattern that can NEVER match:
#      the child's cmdline starts with the shared venv interpreter, so $WT only ever appears AFTER
#      `predict` (in --out_dir), while the pattern requires it before. On a stall it would kill
#      every driver and leave every fold holding a card -- manufacturing the exact orphan-fold
#      failure that cost this campaign ~3.1 card-hours to recover by hand.
#   2. Its hardcoded T0-T3 target subsets were superseded by slices_8, so its relaunch path would
#      have folded the wrong split.
#
# This one is deliberately dumber and cannot damage anything:
#   * It NEVER kills a process and NEVER resets a card. Its only action is to relaunch when the
#     host is provably idle -- zero drivers AND zero predicts alive. There is nothing to race.
#   * It relaunches through abag_xm_tiera_launch.sh, so the slice split stays the committed
#     cost-balanced one and orphaned folds get reconciled first rather than refolded.
#   * Relaunches are capped, so a card that is wedged (the driver now aborts on that -- see
#     DEAD_CARD_STREAK in abag_xm_generate.py) cannot become an infinite relaunch loop.
#
# Deliberately NOT handled: a driver that is alive but stuck. That needs killing, which is what
# made the old watchdog dangerous. A stuck fold is already covered by generate.py's own per-fold
# timeout, which kills only its own child by process group.
#
#   Usage:  nohup setsid scripts/abag_xm_tiera_supervisor.sh [cards] >> ~/abag_xm/logs/supervisor.log 2>&1 &
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARDS="${1:-0 1 2 3}"
LOGDIR=$HOME/abag_xm/logs
POLL="${POLL:-300}"          # 5 min: a fold is 5-35 min, so this notices an idle host quickly
MAX_RELAUNCH="${MAX_RELAUNCH:-12}"
# Label workers, derived from cores. Labelling is the LONGEST phase of this campaign, not a
# background task: the full 492-fold slab is ~72 h at 2 workers on one host against ~10 h of
# remaining generation, and per-label cost has grown ~3x as the remaining targets get larger (qb1
# median 378 s over 80 labels historically, 1050 s over the last 10). The cost of more workers is
# bounded and measured -- ~20% of fold throughput on the 16-core host, nothing detectable on the
# 32-core one -- so slowing generation ~20% costs about 2 h and halving the label phase saves more
# than ten. Spend the idle cores: nproc/8, i.e. 4 on a 32-core host and 2 on 16.
LABEL_WORKERS="${LABEL_WORKERS:-$(( $(nproc) / 8 ))}"
[ "$LABEL_WORKERS" -lt 1 ] && LABEL_WORKERS=1
mkdir -p "$LOGDIR"

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Bracket a character so the pattern cannot match the shell that carries it. Without this a
# `pgrep -f abag_xm_generate.py` inside this script matches the script itself, and the supervisor
# concludes a driver is alive forever (and a kill-based version would shoot itself -- observed
# live on 2026-07-28).
n_drivers(){ pgrep -cf "abag_xm_generat[e].py" 2>/dev/null || echo 0; }
n_predicts(){ pgrep -cf "tt_bio.mai[n] predict" 2>/dev/null || echo 0; }
n_labelloop(){ pgrep -f "abag_xm_labels_loo[p].sh" 2>/dev/null | wc -l; }

relaunched=0
log "supervisor up on $(hostname), cards [$CARDS], poll ${POLL}s, max ${MAX_RELAUNCH} relaunches"
# Claim the host from the fleet dispatcher for as long as this supervisor lives. Without it the
# dispatcher samples the gap between two folds as a free card, takes it, and the next fold dies
# waiting on its own lease -- 25 records on qb1 with a constant 133 s wall clock. The hold
# expires on its own within the hour, so a supervisor that dies does not block the fleet.
bash "$WT/scripts/abag_xm_host_hold.sh" refresh 2>&1 | sed 's/^/    /'
release_hold(){ log "releasing the host hold"
  bash "$WT/scripts/abag_xm_host_hold.sh" release 2>&1 | sed 's/^/    /'; }
# INT and TERM must EXIT, not just run the handler: a bash trap on a signal returns to where it
# was interrupted, so a handler without an exit makes this loop unkillable by anything short of
# SIGKILL -- and SIGKILL skips the release entirely. Observed live on qb2 while swapping
# supervisors: `kill -TERM` ran the handler and the supervisor carried on.
trap 'release_hold; exit 130' INT
trap 'release_hold; exit 143' TERM
trap release_hold EXIT
while :; do
  bash "$WT/scripts/abag_xm_host_hold.sh" refresh >/dev/null 2>&1 \
    || log "WARNING: host hold not refreshed -- the fleet may take a card between folds"
  # Copy the peer's progress file while the peer is still up. A few hundred KB per poll, and it is
  # the difference between a free failover and refolding everything the peer already did -- the
  # choice qb2's hang forced on 2026-07-28, when its 169 records became unreadable.
  bash "$WT/scripts/abag_xm_peer_mirror.sh" >/dev/null 2>&1 \
    || log "note: peer progress mirror not refreshed (peer down?)"
  # Labelling is CPU-only and must overlap generation; its own loop script exists so ~72 h of it
  # does not pile up at the end. It died in the same 2026-07-28 00:16 event that killed the folding
  # drivers and nothing noticed for ten hours, by which point 113 of 181 completed folds were
  # unlabelled. Cheap to restart and safe to have exactly one, so keep one alive. Started, never
  # killed -- same rule as the drivers.
  if [ "$(n_labelloop)" -eq 0 ]; then
    log "labels loop absent -- starting one ($LABEL_WORKERS workers, 2 threads)"
    setsid bash "$WT/scripts/abag_xm_labels_loop.sh" "$LABEL_WORKERS" 2 \
      >> "$LOGDIR/labels_loop.log" 2>&1 < /dev/null &
  fi
  d=$(n_drivers); p=$(n_predicts)
  if [ "$d" -eq 0 ] && [ "$p" -eq 0 ]; then
    if [ "$relaunched" -ge "$MAX_RELAUNCH" ]; then
      log "IDLE but relaunch cap ${MAX_RELAUNCH} reached -- not relaunching again. Something is"
      log "  wrong that a relaunch does not fix; look at gen_card*.log before restarting me."
      sleep "$POLL"; continue
    fi
    # Confirm across two polls before acting, so the gap between one driver exiting and the next
    # being launched by hand is never mistaken for an idle host.
    sleep 20
    d=$(n_drivers); p=$(n_predicts)
    if [ "$d" -eq 0 ] && [ "$p" -eq 0 ]; then
      relaunched=$((relaunched + 1))
      log "IDLE: 0 drivers, 0 predicts -- relaunch ${relaunched}/${MAX_RELAUNCH}"
      bash "$WT/scripts/abag_xm_tiera_launch.sh" "$CARDS" 2>&1 | sed 's/^/    /'
    fi
  fi
  sleep "$POLL"
done
