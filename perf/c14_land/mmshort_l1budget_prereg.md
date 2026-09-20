# Pre-registration: does TT_BIO_MM_SHORT_M_BW move the CIF digest across core grids?

Written and committed BEFORE the arm ran. The screen in `mmshort_grid_screen.json` predicts it
does: `in0_block_w` comes out at 6 on the native 11x10 grid and 4 on 8x8 for `dit_token_768x1536`,
3 against 2 for `768x3072`, 4 against 3 for `768x2304`. Those are different foldings of the same
contraction and bf16 addition is not associative.

The instrument is the release gate's `l1-budget` arm, which is the arm that refused region T. It
folds protenix-v2 at 200 sampling steps on three legs: `native` (11x10 here), `8x8`, and `narrow`,
which is the native grid with the trimul width capped at 128 and is therefore NOT a third grid
class. The discriminating pair is native against 8x8.

Run as: `TT_BIO_MM_SHORT_M_BW=1 ... release_gate.py --model l1-budget`, card 1, `PYTHONPATH` set
to this worktree so the arm scores this tree.

## The two outcomes, both written down now

**A. native and 8x8 return DIFFERENT md5.** The lever makes the structure depend on the core grid.
That is a standing hard stop, it is not scored against the Angstrom bar, and it is the same
mechanism that closed region T. `TT_BIO_MM_SHORT_M_BW` is then **NO-GO as a shipped default**
whatever its +0.047 s reads, and the 18-rep interleaved A/B -- the most expensive thing this row
can ask of a quiet box -- **must not be spent** on it. It may still ship as an opt-in flag
defaulting off, with a docs entry saying enabling it makes the output grid-dependent, exactly as
region T did.

**B. all three legs return ONE md5.** The screen's prediction is refuted for this grid pair: the
configs differ and the numerics do not. The lever is then clear of the hard stop on this box and
the 18-rep A/B is worth a quiet window.

**C. the arm cannot see it.** If `_short_m_proj_program_config` never fires on protenix-v2, the
arm is blind and neither A nor B has been shown. That is a real third outcome and the answer is
then a firing count, not a digest (`eligibility-firing-condition-is-not-a-code-fact`). The gate
does not print `MM_SHORT_M_STATS`, so a green here must not be read as B without it.

## The caveat that stands under every outcome

13x10 is qb1's p150a. On `dit_token_768x768` and `dit_token_1536x768` it folds 12 K tiles where a
p300c folds 8, and the gate's native/8x8 pair agrees on those two shapes. So a single-box gate run
cannot see a p150a-against-p300c disagreement, and outcome B would not clear the lever across
board classes.
