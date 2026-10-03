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
other work still runs on qb2.

## What runs

| unit | what it does |
|---|---|
| `sc26-engine` | the fold service: one warm worker per chip, the stream, the recorded folds |
| `sc26-kiosk` | Firefox in kiosk mode on `http://127.0.0.1:8626/app/`, started once the app answers |
| `sc26-watchdog` | checks the engine, the page and the screen every 10 s and restarts what stopped |

They run in a dedicated login session (`sc26`, a bare sway compositor) instead of the Ubuntu desktop,
so there is no keyring prompt, no update notifier and no window to close. The background behind the
browser is a still of a real fold, so a browser restart never shows a black screen. The Firefox snap
is held at its current version and automatic package upgrades are off for the show; `uninstall`
turns both back on.

## What happens when something fails

| failure | detected by | recovery | what the screen shows |
|---|---|---|---|
| browser crashes | its launcher | Firefox restarts with a fresh profile | the background still, then the app |
| page or browser freezes | no frames drawn, or an unchanged screen for 60 s | the browser is restarted | the frozen frame, then the still, then the app |
| engine dies or hangs | `/status` unanswered for 30 s | systemd restarts it | the last fold, then recorded folds |
| a chip's worker dies | the engine | the worker restarts | that chip's lane says recovering; the other chips keep folding |
| a chip wedges | no progress for 120 s, the worker ignores SIGINT and SIGTERM | both chips on that board are reset with `tt-smi -r` | the board's two lanes say resetting; recorded folds fill in if no chip is left |
| network goes away | nothing to detect | none needed | no change: every model file, font and script is on the box |

Workers are always stopped with SIGINT, then SIGTERM, and never SIGKILL, because a killed worker
leaves its chip unusable until a reset. qb2's chips sit on two boards, chips 0 and 1 on one and
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
