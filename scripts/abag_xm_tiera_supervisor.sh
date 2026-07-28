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
mkdir -p "$LOGDIR"

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Bracket a character so the pattern cannot match the shell that carries it. Without this a
# `pgrep -f abag_xm_generate.py` inside this script matches the script itself, and the supervisor
# concludes a driver is alive forever (and a kill-based version would shoot itself -- observed
# live on 2026-07-28).
n_drivers(){ pgrep -cf "abag_xm_generat[e].py" 2>/dev/null || echo 0; }
n_predicts(){ pgrep -cf "tt_bio.mai[n] predict" 2>/dev/null || echo 0; }

relaunched=0
log "supervisor up on $(hostname), cards [$CARDS], poll ${POLL}s, max ${MAX_RELAUNCH} relaunches"
while :; do
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
