# Running the SC26 booth demo

qb2 boots straight into the demo: no login prompt, full screen, all four chips folding. Nobody at
the booth needs a keyboard. If something breaks, the demo repairs itself and the screen keeps
showing real folds while it does.

## Commands

Run these on qb2, over ssh or at the console, from this directory (`~/sc26/demo/sc26/ops`).

    ./sc26ctl status      what is running, each chip's state and clock, the last watchdog check
    ./sc26ctl restart     restart the engine, the browser and the watchdog
    ./sc26ctl stop        stop the demo; the chips are released to the fleet
    ./sc26ctl start       start it again
    ./sc26ctl logs        follow the demo's logs

    ./sc26ctl install     make the next boot land on the demo (asks for sudo once)
    ./sc26ctl uninstall   go back to the normal Ubuntu login

`sc26ctl` never reboots or powers off the box.

`~/.config/sc26/env` sets which chips the demo uses. `SC26_CHIPS=0,1,2,3` at the booth.
`SC26_CHIPS=` (empty) runs on recorded folds only and takes no chip, which is the setting while
other work still runs on qb2. `SC26_ENGINE_ARGS=--out-of-service 2` takes chip 3 on screen (UMD chip 2)
out of the demo on purpose: leave it out of `SC26_CHIPS` too, and its row says "Out of service". A chip
that hangs does not need this: the engine rests it and brings it back by itself (below).

## What runs

| unit | what it does |
|---|---|
| `sc26-engine` | the fold service: one warm worker per chip, the stream, the recorded folds |
| `sc26-kiosk` | Firefox in kiosk mode on `http://127.0.0.1:8626/app/`, started once the app answers |
| `sc26-watchdog` | checks the engine, the page and the screen every 10 s and restarts what stopped |

They run in a dedicated login session (`sc26`, a bare sway compositor) instead of the Ubuntu desktop,
so there is no keyring prompt, no update notifier and no window to close. The background behind the
browser is a still of a real fold, so a browser restart shows that still, not an empty screen. The Firefox snap
is held at its current version and automatic package upgrades are off for the show; `uninstall`
turns both back on.

## What happens when something fails

| failure | detected by | recovery | what the screen shows |
|---|---|---|---|
| browser crashes | its launcher | Firefox restarts with a fresh profile | sometimes a black frame of about a second, then the background still, then the app |
| page or browser freezes | no frames drawn, or an unchanged screen for 30 s | the browser is restarted | the frozen frame, then the still, then the app (1 restart in 6 flashed black for under half a second) |
| the page leaves the app (Firefox error page) | the page's document is not the app | the watchdog loads the app again once the engine answers | a dark error page for at most one 10 s check |
| engine dies or hangs | `/status` unanswered for 30 s | systemd restarts it | the last fold, then recorded folds |
| a chip's worker dies | the engine | the worker restarts | that chip's lane says recovering; the other chips keep folding |
| a chip's memory fills up (after about 35 min of mixed folds) | the fold fails with out of memory | that worker restarts with empty memory; a visitor's fold is retried | that lane says recovering for about 40 s; the other chips keep folding |
| a chip hangs | tt-metal sees no dispatch progress for 10 s; a chip with no event for 120 s counts too | the worker exits, both chips on that board are reset with `tt-smi -r` (about 40 s), the workers restart and warm up | the board's two lanes say resetting, then warming up |
| a chip keeps hanging (twice within an hour) | the engine | the chip rests for 15 min, then its board is reset if needed and it rejoins by itself. A chip that hangs again soon rests twice as long, up to 4 h | that lane says "Resting after a hang" with the minutes until it is back; its board mate keeps folding |
| network goes away | nothing to detect | none needed | no change: every model file, font and script is on the box |
| the screen is unplugged, or the box boots before it is on | `session/display.sh`, every 2 s | the demo keeps running on an invisible screen and moves onto the real one when it appears | nothing until the screen is back, then the demo |

Workers are stopped with SIGINT, then SIGTERM, and never killed. A worker stuck in the device
ignores both; its board is reset with it still there, which ends the wait so it exits by itself.
If it still has not exited, its chip rests. qb2's chips sit on two boards, chips 0 and 1 on one and
2 and 3 on the other, and a reset always takes both chips of a board.

## Logs

`~/sc26-logs/watchdog.jsonl` has one line per check: the chips, the page's frame rate, whether the
screen is moving, and every minute the browser's and the engine's memory. `~/sc26-logs/engine/`
holds the engine's per-chip logs and `reset.log`.

## Testing it

`chaos.py` injects every failure above on a schedule (browser crash and freeze, engine kill, worker
kill and wedge, a flood of 60 folds, a network drop) and records screenshots of what the screen
showed through each one. `qualify_card.py` runs the fold service on one chip for hours, which is
how a chip is cleared for the booth.
