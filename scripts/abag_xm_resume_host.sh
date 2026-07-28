#!/usr/bin/env bash
# abag_xm_resume_host.sh -- bring this host back into the Tier-A campaign, once.
#
# What a human otherwise has to remember after a reboot or a wedged-host recovery: check the cards
# came back, start EXACTLY ONE supervisor (two both relaunch, and a duplicate's exit trap releases
# the shared fleet hold -- both hit live on 2026-07-28), root it in the right worktree, detach it so
# it survives the ssh session, mirror the peer's progress so a takeover is not needed, and confirm
# the fleet hold actually appeared on the host that reads it. That is a lot of steps to get right
# while looking at a machine that just came back from being hung.
#
#   Usage:  scripts/abag_xm_resume_host.sh [cards]      # cards default "0 1 2 3"
#           --check   report what it would do and exit
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARDS="${1:-0 1 2 3}"
[ "${1:-}" = "--check" ] && { CHECK=1; CARDS="0 1 2 3"; } || CHECK=0
LOGDIR=$HOME/abag_xm/logs
say(){ echo "[resume $(date '+%H:%M:%S')] $*"; }
fail(){ say "ABORT: $*"; exit 1; }

say "worktree $WT"
say "engine tree $(git -C "$WT" rev-parse --short HEAD:tt_bio 2>/dev/null || echo UNKNOWN)"

# A dirty tt_bio/ means every fold this host produces carries unstateable provenance and gets
# refolded. Catch it here rather than after a slice's worth of wasted card time.
if [ -n "$(git -C "$WT" status --porcelain --untracked-files=no -- tt_bio)" ]; then
  fail "tt_bio/ has uncommitted changes -- commit or stash them first, or every fold this host
        produces will be discarded as unpublishable."
fi

# Count the kernel device nodes, not tt-smi's table: tt-smi prints the device list twice, so
# grepping it for the arch reported 8 devices on a 4-card box. The nodes are also the thing that
# actually matters here -- whether a fold can open a card at all. Only the numeric top-level nodes:
# /dev/tenstorrent also holds a by-id/ directory, and globbing expanded it to 10 on the same 4 cards.
ndev=$(ls -1 /dev/tenstorrent 2>/dev/null | grep -c "^[0-9][0-9]*$" || true)
say "$ndev Tenstorrent device node(s) under /dev/tenstorrent"
[ "$ndev" -eq 0 ] && say "WARNING: no device nodes -- every fold will fail at device-open. If the
                          driver is missing after a reboot, rebuild tt-kmd via DKMS first."

# Exclude the `bash -c "... setsid bash supervisor.sh ..."` wrapper: it carries the script name in
# its own cmdline and lingers after spawning, so a plain pgrep counts one supervisor as two and this
# script would refuse to start on a host with none actually running.
n_supervisors(){ pgrep -af "abag_xm_tiera_superviso[r].sh" 2>/dev/null | grep -vc "bash -c" || true; }
running=$(n_supervisors)
# `pgrep -c` prints its count AND exits 1 when the count is zero, so `|| echo 0` appended a
# second zero and this printed "0\n0 driver(s)" on an idle host -- and would have failed an
# integer test the moment anyone compared it numerically. Same miscount class as the two above.
drivers=$(pgrep -cf "abag_xm_generat[e].py" 2>/dev/null || true)
say "already running: $running supervisor(s), $drivers driver(s)"

if [ "$CHECK" = 1 ]; then
  say "--check only; would start a supervisor on cards [$CARDS] if none were running"
  bash "$WT/scripts/abag_xm_host_hold.sh" status || true
  exit 0
fi

if [ "$running" -gt 0 ]; then
  fail "$running supervisor(s) already running. Exactly one must exist -- two both relaunch, and
        stopping a duplicate releases the fleet hold the survivor set. Stop the extras by explicit
        PID first (they honour SIGTERM within a second), then re-run me."
fi

mkdir -p "$LOGDIR" "$HOME/abag_xm/tier_a"
# Mirror the peer while it is reachable: it makes a future failover free instead of a refold of
# everything the peer has already done.
bash "$WT/scripts/abag_xm_peer_mirror.sh" 2>&1 | sed 's/^/    /' || true

say "starting one supervisor on cards [$CARDS]"
cd "$WT" || fail "cannot cd $WT"
setsid nohup bash "$WT/scripts/abag_xm_tiera_supervisor.sh" "$CARDS" \
  >> "$LOGDIR/supervisor.log" 2>&1 < /dev/null &
disown || true

sleep 12
n=$(n_supervisors)
[ "$n" -eq 1 ] || fail "expected exactly 1 supervisor, found $n -- check $LOGDIR/supervisor.log"
say "supervisor up (1)"
bash "$WT/scripts/abag_xm_host_hold.sh" status | sed 's/^/    /'
say "done. Watch: tail -f $LOGDIR/supervisor.log   Progress: scripts/abag_xm_acceptance.py"
