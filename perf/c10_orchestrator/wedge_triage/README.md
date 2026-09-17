# Triaging the 768 aa wedge against the known Blackhole hang class

`c10-size-scaling`'s sixth 768 aa Boltz-2 fold hung a p300c — 101 % CPU in state R, board power at
22 W against the 46 W a fold draws, SIGTERM ignored, ARC alive. Not an error, not an OOM.

The knowledge base carries a standing instruction for exactly this case: *"Action for the next
worker who hits a Blackhole hang (not error, not OOM) on a slice/chunk/reshape op: check this shape
against the sub-tile-last-axis pattern first before treating it as a new mechanism."* This is that
check, done on CPU against the committed 512 aa op census. **Nothing here was run and no root cause
is claimed.**

## The known class

A sub-tile last-axis `slice`/`chunk` of a large TILE-layout DRAM tensor hangs on Blackhole and
returns fine on Wormhole. Piece width decides, not row width. Unfixed upstream, two prior sightings:
BoltzGen's trimul chunk cap above padded seq 2048, and `ttnn.reshape([1,768,512])` killing two
independent runs of `roof_launch/shape_ladder.py`.

## What the fold actually contains

17 ops of the slice/chunk/reshape family are recorded at 512 aa. **Exactly one has a last-axis
extent that is not a whole number of tiles**: `ttnn.permute 1x16x512x48 → 1x16x48x512`, 4,800 calls
per fold. 48 is the per-head channel width, 768/16 — not a tile multiple, but *above* one tile,
where the known class is about widths *below* one. Adjacent, not matching.

## The counter-evidence, which is the stronger point

**The exact shape the second sighting names is in this fold, and it does not hang it.**
`1x768x512` appears as a whole operand in 9,600 calls per 512 aa fold — 4,800 reshapes and 4,800
permutes — and this campaign has completed at least 74 clean 512 aa folds across four concluded
rows. That is roughly **0.7 million executions of the flagged shape with zero hangs.**

So the *shape alone* is not the trigger. That is new information for the second sighting's own open
item, which left "same root cause?" unconfirmed: it weakens shape-alone and strengthens
shape-plus-context — layout, DRAM residency, or allocator state.

## Verdict

**Not matched, and not excluded.** The defining feature of the known class is absent from the
recorded 512 aa shapes, the one near-miss is a 48-wide axis that is above a tile rather than below,
and the campaign's own clean folds argue against the flagged shape being sufficient on its own.

## For the next worker with a chip

- **Capture a 768 aa census first.** This one is 512 aa and every token-scaled extent changes at
  768, so the trigger shape may simply not exist in this table. That is the most useful next step
  and it needs a card.
- Rank by the 17 family shapes here rather than re-walking the whole fold.
- SIGTERM was ignored, so budget for a SIGKILL and a reset, and read the standing rule about killing
  a chip holder before doing it.
- If it does confirm as the same class, file all three sightings upstream together — which is what
  the second-sighting memory already asked for.

## Limits

The census records 60 shapes covering 87 % of calls but only **32 % of bytes**, so a trigger outside
that table is invisible here. One hang in six folds is not a rate. And a shape that appears in a
fold that hung is not thereby the cause.

    python3 wedge_triage.py                      # writes wedge_triage.json
    python3 -m pytest test_wedge_triage.py -q    # 12 controls
