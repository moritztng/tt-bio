## PREDICTION — registered and pushed BEFORE the first arm finished

Written at 2026-09-21, before the bf16-mixed arm was launched and before any `rel_d` from it
existed. Commit that carries it is the first commit on `wk/of3t-trajbar`; nothing in this block is
edited afterwards, and the outcome is recorded below it rather than in place of it.

Reasoning I am committing to: Adam's sqrt(v) normalisation compresses MAGNITUDE error in the
gradient and leaves DIRECTION error largely intact, because the per-coordinate step size is
approximately lr * sign-like and nearly independent of the gradient's scale. A26 says upstream's
own bf16 gradient sits about 10 % away from its float64 gradient in relative L2. A direction error
of that order, partially coherent across steps because the same operator mix is quantised the same
way at every step, should put a twenty-step displacement error in the same order of magnitude
rather than one or two orders below it. So I expect the bar to be LARGE, and I expect our
2.564253e-01 to be a small multiple of it rather than orders above it.

    field                          predicted           interval
    bar rel_d at k = 20            1.5e-01             [6.0e-02, 3.0e-01]
    bar log-log exponent k=2..20   -0.28               [-0.45, -0.10], SUB-LINEAR
    ratio ours / bar at k = 20     1.7x                [0.9x, 4.3x]
    bar d_1                        exactly 0           on both sides, lr(1) = 0
    bar rel_d monotone in k        falling             every rung k = 2..20

    Pre-registered outcome: **outcome 2 of 3 — "our reading is a small multiple of it"**, quoted
    as a multiple in the form D129's 4.388x and D8's 0.747-0.849x are quoted.

    What would falsify each branch, fixed now:
      * outcome 1 (at or under the bar) if ratio <= 1.0x;
      * outcome 2 (small multiple) if 1.0x < ratio <= 10x;
      * outcome 3 (theirs barely moves, ours orders above) if ratio > 10x, or if the bar's
        rel_d at k = 20 is below 1.0e-02.

    None of these is a reason to move a tolerance. PROTOCOL S9 records any amendment with whether
    a number already existed, and one does: 2.564253e-01 at k = 20, `of3t-trajwide`.

VERDICT: PARTIAL — prediction registered, bar not yet run.
