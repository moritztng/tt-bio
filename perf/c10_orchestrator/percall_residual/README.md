# The matmul gap is flat per call, not proportional to work — or the instrument added it

[`../matmul_ceiling/`](../matmul_ceiling/) found the fold's matmuls running ~3.9× below what
arithmetic intensity permits, and concluded that implementation binds rather than DRAM.
"Implementation" is not a lever. This splits it, and the split is decidable from the shapes alone,
because a fixed per-call cost and a uniform rate deficit make **opposite** predictions across a wide
FLOP range.

`residual = modelled µs/call − roofline µs/call`. The recorded shapes span **1152× in FLOPs per
call** — a wide enough lever arm to tell those apart without a device.

| | |
|---|---|
| FLOP/call lever arm | **1152×** |
| residual, all 18 shapes | median 21.3 µs, range 12× |
| residual, the 15 above 10 MFLOP | **median 22.4 µs, sd 7.4, range 4.3×** |
| residual × calls | **2.071 s of a modelled 3.141 s — 66 %** |
| roofline part | 1.070 s |

A rate deficit would have made the residual track the 1152× FLOP range. It doesn't: it spans 4.3×.
**The gap is dominated by a roughly fixed per-call cost of about 22 µs**, and over the recorded
shapes that is two thirds of their time.

The device's own program-launch floor is **1.70 µs**, so the residual is **13× launch** — launch
does not explain it.

## Read this before believing it

**The alternative is that the instrument added it.** The modelled µs/call comes from per-shape rates
measured **on pc**, whose card runs custom 130-core firmware. If pc's harness carried its own fixed
per-measurement overhead of tens of microseconds, the entire residual is a property of that
instrument and not of the fold. Nothing here can separate the two, and this artifact **does not
claim a real per-call cost on qb2**. It makes a prediction instead.

## Pre-registered, for whichever row measures these shapes on qb2

- **If the per-call cost is real**: the qb2 residual is flat in FLOPs at roughly 20 µs, the class
  rate stays near 20 TFLOP/s, and about two thirds of its time sits outside the roofline.
- **If it was pc's instrument**: the residual collapses, large shapes land near their roofline time,
  and the achieved rate rises toward the structural ceiling — staying low only on the genuinely
  DRAM-bound 16-head pair matmuls.

These differ by more than 3× on the achieved rate, so even a coarse measurement separates them.
**Report the residual per shape, not just the class rate** — the class rate alone cannot distinguish
them.

## Limits

Both inputs are modelled: a roofline computed here against a floor computed elsewhere, neither a
measurement of the fold on qb2. 18 shapes, 86 % of the class's calls but 29 % of its modelled floor,
and the residual is **not** extrapolated to the remainder. The three shapes under 10 MFLOP/call are
excluded from the headline spread because their roofline time is under 1 µs so any fixed cost swamps
them by construction; they stay in the table and in the all-shapes statistics. A flat residual is
consistent with several mechanisms — program setup, CB and semaphore configuration, kernel prologue
— and this does not distinguish among them.

    python3 percall_residual.py                      # writes percall_residual.json
    python3 -m pytest test_percall_residual.py -q    # 10 controls
