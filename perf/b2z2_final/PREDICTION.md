# b2z2-final-ceiling — PREDICTED, before any arithmetic was run

Written after reading `state/b2z2/CONTEXT.md` and all 2825 lines of `state/b2z2/FINDINGS.md`, and
before `closing.py` existed. The commit order is the evidence.

## PREDICTED

1. **Single-processor ceiling: 1.85x-2.00x.** I expect `b2z2-redteam-ceiling`'s 1.954x-2.135x to
   come down, because its optimistic rung applies the *Pairformer block's* movement-free multiplier
   to the diffusion sampler, and `b2z2-bh-tile-census` says the sampler's CB-wait columns are blank
   on every architecture. I expect the top of the range to land just under 2x.
2. **The route table survives the double-counting check.** The bounds are nested rather than
   additive -- every trunk lever attacks the same input-tile wait the trunk-only bound already
   zeroes -- so nobody has stacked them, and the table reads as a set of sub-bounds. I expect to
   find at most a presentational problem, not an arithmetic one.
3. **The headline moves by less than 2 %.** Wave 2's composed measurement is 1.00x inside a
   1.00229x A/A floor; there is no lever to re-price, so I expect re-derivation to confirm the
   published numbers and change nothing.

## FALSIFIER

If re-derivation moves the campaign's headline by more than 2 %, that is the finding and it leads.
The specific things that would do it: a denominator used with a numerator from a different arm; a
ceiling rung quoted against a base it was not derived from; or a parity verdict resting on an
instrument the published cell says cannot read this fixture.
