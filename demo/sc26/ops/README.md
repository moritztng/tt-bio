# Running the SC26 booth demo

qb2 boots straight into the demo: no login prompt, full screen, all four chips folding. Nobody at
the booth needs a keyboard. If something breaks, the demo repairs itself and the screen keeps
showing real folds while it does.

## Commands

Run these on qb2, over ssh or at the console, from this directory (`~/sc26/demo/sc26/ops`).

    ./sc26ctl health      HEALTHY, REPAIRING ITSELF or NOT HEALTHY, in plain words
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
out of the demo on purpose: leave it out of `SC26_CHIPS` too, and its row says "Out of service".

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

The rule is that the screen never shows an error, a blank frame or a frozen one for longer than
the recovery takes. Every row was injected and watched (`chaos.py`); the times are measured.

| failure | detected by | recovery | what the screen shows |
|---|---|---|---|
| a chip hangs mid-fold (stops taking work) | tt-metal's dispatch timeout in the worker (10 s without progress), or the engine's stall limit (120 s with no event) | the board's workers are stopped and the board (chips 0,1 or 2,3) is reset; a chip that hangs twice in an hour rests, then rejoins by itself | the board's two lanes say resetting, then warming up; the other board keeps folding; recorded folds fill the stage if no chip is left |
| a chip dies and does not come back after a reset | the reset's check that the chip answers | the chip rests, 15 min doubling to 4 h, and is tried again | its lane says it is resting and when it is back |
| a chip's worker dies | the engine | the worker restarts; a visitor's fold moves to another chip | that lane says recovering, then warming up |
| a chip's memory fills up (about 35 min of mixed folds) | the fold fails with out of memory | that worker restarts with empty memory | that lane says recovering for about 40 s |
| the engine dies | the watchdog (3 unanswered checks, 30 s) and systemd | the old engine stops its chip workers, then a new one starts: answering again 41 to 111 s after the kill (soak, 5 kills); the page reconnects by itself | the last fold keeps turning, then recorded folds while the chips warm up |
| the engine is alive but stuck | it stops telling systemd it is alive (every 2 s); the watchdog sees /status unanswered | systemd ends it and starts it again: answering again 100 to 111 s after the freeze (soak, 5 freezes) | as above, with lanes saying "no word" until then |
| the browser crashes | its launcher | Firefox starts again with a fresh profile, once the app answers | the background still (a real fold), then the app |
| the page or browser freezes | no frames drawn for two checks (20 s), or an unchanged screen for 30 s | the browser is restarted | the frozen frame, then the still, then the app |
| the page leaves the app | its document is not the app | the watchdog loads the app again once it answers; an error page while the app is down restarts the browser, whose launcher shows the still until the app answers | at most one 10 s check of the wrong page |
| the GPU context is lost | the page itself | the page reloads once the app answers | under a second of the background colour |
| the page's stream goes silent | the page (6 s); the watchdog backs it up at 60 s | the page reconnects; the watchdog reloads a page that does not, but only once the app answers (loading it while the engine restarts showed Firefox's "Unable to connect" page in the 10-06 soak) | the stage keeps showing what it has |
| the compositor (sway) freezes | screenshots time out three checks in a row | the watchdog restarts sway; the session brings it back with the browser | the frozen frame for up to 40 s, then the still, then the app |
| the compositor crashes | the session's loop | sway starts again in 2 s, the browser follows | the still within 5 s, the app in about 30 s |
| the screen is unplugged, or the box boots before it is on | `session/display.sh`, every 2 s | the demo keeps running on an invisible screen and moves onto the real one when it appears | nothing until the screen is back, then the demo |
| the network goes away | nothing to detect | none needed | no change: every model file, font and script is on the box |
| the clock jumps (NTP at the booth) | nothing to detect | every timeout and the chips' heartbeat check run on monotonic clocks (`ops/tests/clock_jump.py`: a 1 h step either way changes no lane) | at most one fold's "N min ago" is off until it is folded again |
| the hall is warm and the chips throttle | each fold records its clock; `sc26ctl health` names a chip whose last fold ran under 1200 MHz | none: the chip's firmware protects it | folds take longer and the stopwatch shows the real time |
| nobody touches the kiosk for hours | | none needed: the attract loop folds the gallery proteins without visitors, and the soak runs with no visitor input | the demo, unchanged |
| memory, file descriptors, GPU memory or disk run out | the watchdog samples each every minute; `sc26ctl health` warns under 16 GB memory or 5 GB disk | none should be needed: the 24 h soak measures each one, and every log is bounded | |
| logs grow for days | the watchdog, every minute | any log past 64 MB is cut to its last 16 MB, in place | |
| the host kernel locks up (seen once, 2026-10-05 19:18Z) | the hardware watchdog: systemd stops feeding /dev/watchdog0 and the board reboots after 2.5 min | the box reboots and starts the demo by itself; measured: hung 19:18:52Z, back 19:21:40Z, demo units running 19:21:42Z, first live fold 19:27:00Z | a frozen screen for about 3 min, a black screen while it boots, then the still and recorded folds; live folds about 8 min after the hang |
| a power cut | | the box boots into the demo by itself when power returns and the button is pressed | the still within a minute, live folds in about 1½ |

Workers are stopped with SIGINT, then SIGTERM. qb2's chips sit on two boards, chips 0 and 1 on one
and 2 and 3 on the other, and a reset always takes both chips of a board. Nothing here powers the
box off.

## Logs

`~/sc26-logs/watchdog.jsonl` has one line per check: the chips, the page's frame rate and stream,
whether the screen is moving, and every minute the memory and open files of the engine, each chip
worker and the browser, the GPU's memory, free disk and the logs' size. `curves.py` draws them.
`~/sc26-logs/engine/` holds the engine's per-chip logs and `reset.log`.

## Testing it

`chaos.py` injects the failures above on a schedule (browser crash and freeze, engine kill and
freeze, worker kill and wedge, compositor freeze and crash, a display unplugged, a flood of 60
folds, a network drop) and records screenshots of what the screen
showed through each one. `qualify_card.py` runs the fold service on one chip for hours, which is
how a chip is cleared for the booth.
