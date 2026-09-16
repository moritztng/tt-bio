# Holding the card clock

A Boltz-2 fold at 512 residues blocks on the host around 200 times. Between those syncs the chip
has nothing to do, the Blackhole clock governor sees an idle card and drops it to its 800 MHz
floor, and the next burst of work starts from there. The card is not thermally or power limited
while this happens; it is simply being asked to be idle several hundred times a second.

`TT_BIO_AICLK=<MHz>` holds the clock for the life of the card instead. It is off by default.

## What it is worth

qb2, p300c (board `...4103`, node 1), Boltz-2 `cdk2x2` at 512 aa, 200 sampling steps, 3 recycles.
Eight timed folds per arm plus one discarded warmup, arms alternating fold by fold inside one
process with one device open and one model load, so nothing slow-moving can line up with an arm.
Harness `perf/b2z2_aiclk_pin/pin_ab.py`, raw data `perf/b2z2_aiclk_pin/out/force_ab_qb2c1.json`.

| | median | folds (sorted) | AICLK mean | burst fraction | card power |
|---|---|---|---|---|---|
| governor | 18.990 s | 15.933 … 21.130 | 1021.4 MHz | 0.39 | 70.8 W |
| held at 1350 | 14.987 s | 14.745 … 15.884 | 1347.9 MHz | 0.996 | 100.7 W |

**1.267x on the median.** The held arm's own clock trace is the check that this is real and not a
drifting box: every held fold sat at 0.98-1.00 burst fraction, against a control arm whose minimum
never left 800 MHz.

Two things make 1.267x a floor rather than an estimate:

- Each held fold leaves the card hot and the governor boosted going into the control fold that
  follows it. The control arm walks from 21.130 s (876 MHz) in its first fold down to 15.933 s
  (1206 MHz) later in the same session. A cold governor against a held clock is 21.130 / 14.987,
  about 1.41x.
- The box was busy with another job on the neighbouring card throughout. Co-tenant work on a p300
  keeps the governor boosted, which again flatters the control arm.

So the win is largest exactly where it matters most for a hosted service: a single fold arriving at
an otherwise idle card, which is the case where the governor has sagged furthest.

The same measurement through the shipped `TT_BIO_AICLK` path rather than the harness, one fold each
way on a quiet card: **20.167 s unset against 15.053 s at 1350, so 1.340x**, with the held fold
averaging exactly 1350.0 MHz and needing no re-assert.

## Accuracy

Unchanged, and not in a hand-waving sense. Over two sessions and 33 folds the held arm never wrote
a structure the governor arm did not also write; in the second session every fold of both arms was
bit-identical. The first session's governor arm wrote three different structures against the held
arm's one, but that is the run-to-run nondeterminism Blackhole already shows at this size, and one
session is not enough to claim that a steady clock buys determinism. pLDDT across both arms of
session one spans 0.842325 to 0.845919, with the held arm sitting on 0.845919, a value the governor
arm also produced.

Nothing about the model changes: same sampling steps, same recycles, same kernels, same tensors.
Only the rate the card retires them.

## What it costs

- **Power**: about 56-57 W per fold on the governor against 98-103 W held, so roughly +30 W
  averaged over a fold and +42 W while one is actually running. `power1_input` is per chip, so on a
  two-chip p300 board the two cards are budgeted separately.
- **Heat**: 60.0-65.8 C on the governor against 62.1-65.5 C held. Not the limiting factor.
- **Reliability**: unmeasured, and not claimed either way. The box these numbers come from has an
  unrelated PCIe fault that reset it roughly every 15-60 minutes throughout, which swamps anything
  a ten-minute run could show about whether a held clock shortens card life.

## Why it is opt-in and not a default

The mechanism is ARC message `0x33`, `FORCE_AICLK`, sent through tt-kmd's SMC message queue. The
driver does not document it. It is also chip state rather than fd state: it outlives the fd it was
sent on *and* the process that sent it, so a run killed without releasing leaves the card at
1350 MHz and about 71 W idle against 30-34 W, indefinitely. `tt_bio/aiclk.py` releases on the
normal path, on `cleanup()`, on `atexit` and on SIGTERM/SIGHUP/SIGINT, but a `SIGKILL` can still
strand it. Defaulting that on, for every user, on hardware whose power budget we do not own, is not
a trade to make silently.

It is Blackhole-only. `0x33` is a Blackhole ARC opcode and the module refuses to send it anywhere
else rather than guess at another architecture's power-management messages.

## The knob is not sticky, so it is watched

Any *legacy* open of the same chip makes tt-kmd recompute an aggregated power state across every
open fd and re-send it (`chardev.c:545`). A legacy open is anything without `O_APPEND`, which
includes every UMD open, and tt-kmd initialises it to `TT_POWER_FLAG_ALL & ~TT_POWER_FLAG_MAX_AI_CLK`
(`chardev.c:987`) — explicitly *minimum* AI clock. So another process touching the card can put the
clock back under the governor mid-fold. Measured rate during a Boltz-2 run: one clear in thirteen
folds. `tt_bio/aiclk.py` polls at 4 Hz and re-sends the force when the clock is found more than
50 MHz under target, so a clobber costs a quarter second rather than a silently slow fold.

## Two knobs that do not work

- **`TENSTORRENT_IOCTL_SET_POWER_STATE` with `TT_POWER_FLAG_MAX_AI_CLK`.** The documented API, and
  the one to prefer if it worked: tt-kmd ORs `power_flags` across all open fds, so the bit would
  survive a co-tenant open and drop automatically on close, with none of `FORCE_AICLK`'s leak
  hazard. Firmware 19.11 accepts it and the clock does not move. It was first written off against a
  saturating matmul, which proves nothing either way because that load is pinned at the governor's
  loaded ceiling regardless; it has since been re-measured against a real fold, the fixture that
  actually has the idle gaps, and it is still flat. Raw data
  `perf/b2z2_aiclk_pin/out/maxclk_ab_qb2c1.json`, harness `pin_ab.py --knob maxclk`.
- **`AICLK_GO_BUSY`.** Already sent by UMD at every device open, so there is nothing to add.
