#!/bin/bash
# Arm + supervise the deep-N labeler on qb1 once the galaxy harvest has landed.
#
# Why this exists: watch_galaxy_drain_harvest.sh ends by tg-messaging "no labeler is
# running -- launch abag_xm_deepn_label.py on qb1 next". Nothing acts on that message.
# The harvest lands ~2565 unlabeled folds at 07:00Z with no process aware of them, and
# labelling is the ~9 h critical path before any 512 number exists. Same class as the
# p31b/p32 chain and the unarmed harvest: a stage whose trigger is a human.
#
# Three things this gets right that the hand-written runbook command did not:
#
#  1. --base MUST be the galaxy tree. The labeler's default base is ~/abag_xm/deepn,
#     whose <model>/ dirs are the legacy n64 tree (32 dirs each, fully labelled). Every
#     harvested fold lands in ~/abag_xm/deepn/galaxy/<model>/ (5874 dirs). The bare
#     command scans the wrong tree, finds nothing, and idles forever -- indistinguishable
#     from a labeler that finished. Every p27-era run used base=.../galaxy (first line of
#     ~/abag_xm/deepn/label_galaxy_bz.log).
#  2. CAMPAIGN_DONE is resolved against --base, so with the galaxy base the marker is
#     galaxy/CAMPAIGN_DONE. Written anywhere else the labeler never exits.
#  3. Stale .label_lock files strand their dirs permanently. The lock holds the labeler's
#     pid, so at the one moment we relaunch -- zero labelers alive -- every lock on disk
#     is provably stale and safe to clear. Without this a SIGKILL loses up to $WORKERS
#     dirs silently, which the completeness gate then reports as missing cells.
#
# Runs on pc, setsid-detached, ppid 1. Supervises: relaunches a died labeler up to
# MAXLAUNCH times, exits when the campaign marker is up and no labeler is left.
#
# usage: watch_harvest_label.sh [gate_run]        default gate: p31
set -u
GATE=${1:-p31}
QB=${QB:-qb1}
QWT=${QWT:-.coworker/wt/abag-n512}          # qb1-relative, holds this branch
WORKERS=${WORKERS:-12}                       # qb1 is 32-core, co-tenanted; nice -15 yields
POLL=${POLL:-300}
MAXLAUNCH=${MAXLAUNCH:-5}
STATE=$HOME/.coworker/state/deepn_harvested_$GATE
LOGD=$HOME/abag_xm/deepn/logs; mkdir -p "$LOGD"
LOG=$LOGD/label_arm.log
RLOG='$HOME/abag_xm/deepn/logs/label_n512.log'
S="ssh -o BatchMode=yes -o ConnectTimeout=20 $QB"

say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

# pgrep -f matches the ssh wrapper's own argv, so the bracket form is mandatory here:
# the wrapper carries the literal "[a]bag_xm_deepn_label", which the pattern never matches.
alive() { [ -n "$($S 'pgrep -f "[a]bag_xm_deepn_label.py" | head -1' 2>/dev/null)" ]; }
done_marker() { $S "test -f \$HOME/abag_xm/deepn/galaxy/CAMPAIGN_DONE" 2>/dev/null; }

launches=0
say "armed: gate=$STATE qb=$QB workers=$WORKERS (labeler base = galaxy tree)"
while :; do
  if done_marker && ! alive; then
    say "CAMPAIGN_DONE up and no labeler alive -- supervision over, exit"
    exit 0
  fi
  if [ -f "$STATE" ] && ! alive; then
    if [ "$launches" -ge "$MAXLAUNCH" ]; then
      say "FATAL: $MAXLAUNCH launches and the labeler will not stay up -- giving up"
      "$HOME/.coworker/tg.sh" send "abag-xm deepn: the qb1 labeler died $MAXLAUNCH times; PHASE 3 is stalled with no labels. See $QB:~/abag_xm/deepn/logs/label_n512.log."
      exit 1
    fi
    # Provably stale: nothing is labelling right now, so no lock has a live owner.
    nlock=$($S 'find $HOME/abag_xm/deepn/galaxy -maxdepth 3 -name .label_lock -delete -print 2>/dev/null | wc -l')
    # No git here on purpose: the supervisor must not mutate the worktree the analysis
    # steps read. Keep $QWT current by hand; the labeler itself has not changed since it
    # was written and resolves WT from its own location.
    $S "cd \$HOME/$QWT && \
        setsid nohup nice -15 python3 -u scripts/abag_xm_deepn_label.py \
          --base \$HOME/abag_xm/deepn/galaxy --workers $WORKERS \
          </dev/null >>$RLOG 2>&1 &" </dev/null >/dev/null 2>&1
    sleep 15
    launches=$((launches + 1))
    if alive; then
      say "labeler launched (attempt $launches, workers=$WORKERS, cleared $nlock stale locks)"
      [ "$launches" -eq 1 ] && "$HOME/.coworker/tg.sh" status "abag-xm deepn: harvest $GATE landed, labeler running on qb1 over the galaxy tree (workers $WORKERS). This is the ~9 h critical path to the 512 numbers."
    else
      say "launch attempt $launches did not take; retrying in ${POLL}s"
    fi
  fi
  sleep "$POLL"
done
