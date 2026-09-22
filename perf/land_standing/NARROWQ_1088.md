# narrow-q at 1088 aa: the falsifier rung fires and does nothing

`TT_BIO_TRIATT_NARROW_Q_FALLBACK`, rf3 at 1088 aa, qb2 card 0 (p300c), board-pair sibling card 1
verified holding no fd for both runs, AICLK sampled DURING every leg with an unbroken run at or
above 1200 MHz containing each timed fold.

## Why this rung was worth a window

1088 is not a multiple of 256, so the policy fires there, and L1 refuses more configs at the
longer length. If a narrower q_chunk ever costs more than the fused path saves, this is where it
shows: the kernel re-reads all of K and V once per q-chunk, so narrowing trades re-reads for
keeping the fused path. 896 alone would have been a default flipped on one length, which is the
standing one-size defect.

## Result: inert, and not a regression

    rung 1088   pairs 2 (both arm orders)
      lever ON  160.8, 161.0   median 160.90 s
      shipped   161.1, 161.2   median 161.15 s
      A/A floor +0.124 %   A/B +0.155 %   effect / floor 1.25x

Read this as **inert**, not as a 0.25 s win. The script's binary verdict says PASS because the
arms happen not to overlap, but an effect at 1.25x its own floor is not a result, and four legs
spanning 0.4 s total is the honest description. What the rung DOES establish is the thing it was
run to establish: **the sign is not adverse.** The lever does not cost anything at the length
where it had the best chance to.

Compare the live cell, scored the same way on the same card:

    rung  896   +9.5000 s   1.1005x   effect / floor 15.83x
    rung 1088   +0.2500 s   1.0016x   effect / floor  1.25x

## So the lever fires in two places and pays in one

This is `a-lever-can-fire-and-be-inert`. The policy firing is a property of divisibility; whether
it changes anything depends on whether L1 refused every wider dividing chunk, which at 1088 it
evidently did not in the way that matters. The reach audit over 142 baseline censuses had already
left rf3 at 896 and 1088 as the only dark-and-firing cells; this run removes 1088 from the list of
places the default flip would change a fold's cost.

## Accuracy at this rung

**One distinct CIF sha256 across all 6 legs**, both arms, both processes:

    4f74fda5d7194490ecc745b481735b8c

Bit-exact, consistent with 896 aa's 12 legs and with the mechanism: q_chunk splits output rows and
the online softmax reduces over k, so no reduction order changes.

## What is still owed before the default flips

The full release gate at the branch tip, arm by arm, plus the local-main check and the test run.
None of those needs a quiet box -- `scripts/release_gate.py` has no timing arm.
