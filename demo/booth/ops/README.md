# Running the TT-Bio booth demo

qb2 boots straight into the demo: no login prompt, full screen, all four chips folding. Nobody at
the booth needs a keyboard. If something breaks, the demo repairs itself and the screen keeps
showing real folds while it does.

## Commands

Run these on qb2, over ssh or at the console, from this directory (`~/tt-bio-booth/demo/booth/ops`).

    ./boothctl health      HEALTHY, REPAIRING ITSELF or NOT HEALTHY, in plain words
    ./boothctl status      what is running, each chip's state and clock, the last watchdog check
    ./boothctl restart     restart the engine, the browser and the watchdog
    ./boothctl stop        stop the demo; the chips are released to the fleet
    ./boothctl start       start it again
    ./boothctl logs        follow the demo's logs

    ./boothctl install     make the next boot land on the demo (asks for sudo once)
    ./boothctl uninstall   go back to the normal Ubuntu login

`boothctl` never reboots or powers off the box.

`~/.config/booth/env` sets which chips the demo uses. `BOOTH_CHIPS=0,1,2,3` at the booth.
`BOOTH_CHIPS=` (empty) runs on recorded folds only and takes no chip, which is the setting while
other work still runs on qb2. `BOOTH_ENGINE_ARGS=--out-of-service 2` takes UMD chip 2
out of the demo on purpose: leave it out of `BOOTH_CHIPS` too. The screen then lists only the chips that
fold, numbered by place (UMD 0, 1, 3 read as chips 1, 2, 3, here and in `boothctl health`), and the
hardware view counts them ("Three chips, live"), so no row stands empty and no number is missing. A chip
that hangs does not need this: the engine rests it and brings it back by itself (below).

## What runs

| unit | what it does |
|---|---|
| `booth-engine` | the fold service: one warm worker per chip, the stream, the recorded folds |
| `booth-kiosk` | Firefox in kiosk mode on `http://127.0.0.1:8626/app/`, started once the app answers |
| `booth-watchdog` | checks the engine, the page and the screen every 10 s and restarts what stopped |

They run in a dedicated login session (`booth`, a bare sway compositor) instead of the Ubuntu desktop,
so there is no keyring prompt, no update notifier and no window to close. The background behind the
browser is a still of a real fold, so a browser restart shows that still, not an empty screen. The Firefox snap
is held at its current version and automatic package upgrades are off for the show; `uninstall`
turns both back on.

## What happens when something fails

The rule is that the screen never shows an error, a blank frame or a frozen one for longer than
the recovery takes. Every row was injected and watched (`chaos.py`); the times are measured.

| failure | detected by | recovery | what the screen shows |
|---|---|---|---|
| a chip hangs | tt-metal sees no dispatch progress for 10 s; a chip with no event for 120 s counts too | the worker exits, both chips on that board are reset with `tt-smi -r` (about 40 s), the workers restart and warm up | the board's two lanes say resetting, then warming up |
| a chip keeps hanging (twice within an hour) | the engine | the chip rests for 15 min, then its board is reset if needed and it rejoins by itself. A chip that hangs again soon rests twice as long, up to 4 h | that lane says "Resting after a hang" with the minutes until it is back; its board mate keeps folding |
| a chip dies and does not come back after a reset | the reset's check that the chip answers | the chip rests, 15 min doubling to 4 h, and is tried again | its lane says it is resting and when it is back |
| a chip's worker dies | the engine | the worker restarts; a visitor's fold moves to another chip | that lane says recovering, then warming up |
| a chip's memory fills up (about 35 min of mixed folds) | the fold fails with out of memory | that worker restarts with empty memory | that lane says recovering for about 40 s |
| the engine dies | the watchdog (3 unanswered checks, 30 s) and systemd | the old engine stops its chip workers, then a new one starts: answering again 41 to 111 s after the kill (soak, 5 kills); the page reconnects by itself | the last fold keeps turning, then recorded folds while the chips warm up |
| the engine is alive but stuck | it stops telling systemd it is alive (every 2 s); the watchdog sees /status unanswered | systemd ends it and starts it again: answering again 100 to 111 s after the freeze (soak, 5 freezes) | as above, with lanes saying "no word" until then |
| the browser crashes | its launcher | Firefox starts again with a fresh profile, once the app answers | the background still (a real fold), then the app |
| the page or browser freezes | no frames drawn, or no answer, for two checks (10 s apart), or an unchanged screen for 30 s | the browser is restarted | the frozen frame for up to 27 s (soaks, 6 freezes), then the still, then the app |
| the page leaves the app | its document is not the app | the watchdog loads the app again once it answers; an error page while the app is down restarts the browser, whose launcher shows the still until the app answers | at most one 10 s check of the wrong page |
| the GPU context is lost | the page itself | the page reloads once the app answers | under a second of the background colour |
| the page's stream goes silent | the page (6 s); the watchdog backs it up at 60 s | the page reconnects; the watchdog reloads a page that does not, but only once the app answers (loading it while the engine restarts showed Firefox's "Unable to connect" page in the 10-06 soak) | the stage keeps showing what it has |
| the compositor (sway) freezes | a screenshot takes over 2 s, checked again 1 s later, three times | the watchdog restarts sway; the session brings it back with the browser | the frozen frame for about 15 s, a black screen while sway starts again (not measured: qb2 has no monitor), then the still, then the app |
| the compositor crashes | the session's loop | sway starts again at once (after 2 s if it died within 10 s of starting), the browser follows | a black screen while sway starts again (the empty text console; not measured, qb2 has no monitor), then the still, then the app within 5 to 20 s |
| the screen is unplugged, or the box boots before it is on | `session/display.sh`, every 2 s | the demo keeps running on an invisible screen and moves onto the real one when it appears | nothing until the screen is back, then the demo |
| the network goes away | nothing to detect | none needed | no change: every model file, font and script is on the box |
| the clock jumps (NTP at the booth) | nothing to detect | every timeout and the chips' heartbeat check run on monotonic clocks (`ops/tests/clock_jump.py`: a 1 h step either way changes no lane) | at most one fold's "N min ago" is off until it is folded again |
| the hall is warm and the chips throttle | each fold records its clock; `boothctl health` names a chip whose last fold ran under 1200 MHz | none: the chip's firmware protects it | folds take longer and the stopwatch shows the real time |
| nobody touches the kiosk for hours | | none needed: the attract loop folds the gallery proteins without visitors, and the soak runs with no visitor input | the demo, unchanged |
| memory, file descriptors, GPU memory or disk run out | the watchdog samples each every minute; `boothctl health` warns under 16 GB memory or 5 GB disk | none should be needed: the 24 h soak measures each one, and every log is bounded | |
| logs grow for days | the watchdog, every minute | any log past 64 MB is cut to its last 16 MB, in place | |
| the host locks up (three times in the soaks: 2026-10-05 19:18Z, 10-06 17:36Z and 10-07 12:24Z, each within a minute of the engine opening its chips again) | a chip card drops off the PCIe bus first: its hwmon power and current read 0xFFFFFFFF. On qb2 that reading came before every unclean death since 2026-09-15 (66 of 66) and never without one, 32 to 197 s ahead; the kernel logs nothing. `booth-card-sentinel` (root) reads every card every 2 s; the hardware watchdog (2.5 min) stays as the backstop | the sentinel reboots the box after 3 all-ones reads in a row (s, u, b through sysrq: a clean shutdown would stop the engine, which touches the dead card); the box boots into the demo by itself. Without the sentinel, 10-07: card off the bus 12:23:52Z, screen at 3 to 5 fps, last sign of life 12:24:56Z, rebooted 12:27:51Z, the app at 62 fps 12:28:19Z | with the sentinel (not yet seen on a real event): up to 10 s at a few fps, a black screen for about 45 s while it boots, then the still and recorded folds. Without it: 1 to 5 fps for about a minute, a frozen screen for about 3 min, black for about 45 s; live folds 8 to 9 min after the hang |
| a power cut | | the box boots into the demo by itself when power returns and the button is pressed | the still within a minute, live folds in about 1½ |

Workers are stopped with SIGINT, then SIGTERM, and never killed. The hung chip's own worker may
still be there when its board is reset; the reset ends its wait and it exits by itself. A board
mate that is still working is never reset under: resetting a chip in use hard-hung qb2 once. The
mate gets up to 5 min to finish its step or warm-up; if it is still working, the board is left
alone and the hung chip rests. qb2's chips sit on two boards, chips 0 and 1 on one and
2 and 3 on the other, and a reset always takes both chips of a board. Nothing here powers the
box off.

## Logs

`~/booth-logs/watchdog.jsonl` has one line per check: the chips, the page's frame rate and stream,
whether the screen is moving, and every minute the memory and open files of the engine, each chip
worker and the browser, the GPU's memory, free disk and the logs' size. `curves.py` draws them.
`~/booth-logs/engine/` holds the engine's per-chip logs and `reset.log`.

## Testing it

`chaos.py` injects the failures above on a schedule (browser crash and freeze, engine kill and
freeze, worker kill and wedge, compositor freeze and crash, a display unplugged, a flood of 60
folds, a network drop) and records screenshots of what the screen
showed through each one. `qualify_card.py` runs the fold service on one chip for hours, which is
how a chip is cleared for the booth.
