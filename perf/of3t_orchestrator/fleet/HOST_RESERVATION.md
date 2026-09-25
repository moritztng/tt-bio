# Reserving a host's memory for one run, and the one way to get it wrong

Found 2026-09-25 by `of3t-orchestrator` (pass 497) after `of3t-stepqb2` lost the campaign's last
measurement twice to host memory and concluded that the fix was *"a reservation, not a smaller
scope"* while also concluding that no reservation existed. It does exist, in `fleet.sh`, and
nothing pointed at it.

## The mechanism

`state/<host>-hold` holds a unix epoch. `host_blocked()` (`fleet.sh:696-706`) refuses every new
dispatch to that host until the epoch passes, then deletes the file itself. It does **not** touch
rows already running, so it drains a host rather than clearing it. That is the right shape for a
memory reservation: what usually kills a large run is co-tenants launching beside it, not one
large neighbour already there.

```sh
echo $(( $(date +%s) + 1800 )) > /home/moritz/.coworker/state/pc-hold   # 30 min
# ... run ...
rm -f /home/moritz/.coworker/state/pc-hold                              # release, do not wait
```

## The trap

The check is:

```sh
u=$(cat "$D/state/$h-hold" 2>/dev/null || echo 0)
if [ "${u:-0}" -gt "$(date +%s)" ] 2>/dev/null; then ... else rm -f "$D/state/$h-hold"; fi
```

It reads the **whole file**. Add a second line — a reason, a slug, a timestamp — and the integer
comparison errors, the error is swallowed by the `2>/dev/null`, and control falls to the `else`
branch, **which deletes your hold**. Tested both ways on a scratch file before this was written: a
bare epoch is honoured, an epoch plus a comment line self-deletes on the next 2-minute tick.

**So the file is one bare integer and nothing else.** The reason goes in your state doc.

`state/notbefore/<slug>` forgives a trailing audit block because it parses the first whitespace
token. This one does not. Two markers, two parsers, and the one that does not forgive you is the
one whose failure is silent — you get dispatched onto a host you thought you had reserved, and
nothing anywhere says why.

## Cost, and why it is capped

`fleet.log` prints `ANOMALY: <host>-hold set to <epoch> ... unknown writer` for any delta over
300 s, because `wake_host()` only ever writes now+240. That line is yours. pc carries up to 12
`card=cpu` rows under `MAX_CPU_ROWS_PER_HOST=12`, so a hold stalls other campaigns for its whole
duration: take it immediately before the run that needs it, cap it at 30 minutes, and release it
by hand rather than letting it expire.
