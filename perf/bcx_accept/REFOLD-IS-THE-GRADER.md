# Passing every design stage is not acceptance, and my own instruments said it was

Found 2026-09-26 03:4xZ while pooling the adopted `traj_off_s3` arm into the denominator.

## The defect

BindCraft 2 prints one line when the MPNN refold ensemble finishes scoring a trajectory:

    3 of 10 redesigns passed, keeping the best 1 by i_pDAE: candidate 2 (0.31)
    0 of 10 redesigns passed

Both forms are printed. `stage_profile.py` and `rate_ledger.py` each matched
`\d+ of \d+ redesigns passed` and wrote `ACCEPTED` without reading the leading count, so a
trajectory whose ten refolds were **all rejected** was recorded as an accepted binder.

It stayed latent because no arm had produced the `0 of` form until now: every trajectory that
reached the refold ensemble before today kept at least one candidate. `traj_off_s3` is the first
arm where the ensemble rejected a trajectory outright, and it did so three times, so the unpatched
instruments read **4 accepted** on that arm against the **1** its own `.campaign_state.json` and
`2_Refolded/!_Refolded.csv` both record. Across the campaign it would have published 6 acceptances
where there are 3.

Fixed in both files: read the count, and record a zero-keep as the terminal stage it is,
`validation [0 of 10]`.

## The finding underneath it, which is worth more than the fix

`validation` is a terminal stage, it is where the device loses most often among trajectories that
clear the gradient loop, and **the gradient loop's own metrics do not predict it.**

    draw   screen      refine      anneal      harden      mutate      refold verdict
    l93    0.81/0.92   0.78/0.88   0.82/0.91   0.72/0.84   0.76/0.85   3 of 10 kept -> ACCEPTED
    l73    0.82/0.91   0.84/0.91   0.84/0.92   0.84/0.90   0.75/0.86   0 of 10
    l151   0.89/0.94   0.87/0.93   0.88/0.94   0.79/0.87   0.83/0.91   0 of 10
    l82    0.84/0.92   0.81/0.93   0.84/0.93   0.83/0.92   0.77/0.90   0 of 10

`l151` carries the highest screen i_pTM of any trajectory in the campaign, device or reference,
and the best mutate pair on this arm. All ten of its refolds failed on `i_pTM`. `l93` is the
weakest of the four at harden and it is the one that accepted. Every one of these four passed all
five design stages, so a stage table alone cannot rank them.

Two consequences:

1. **A stage-profile row is not an acceptance prediction.** The stage numbers come from the design
   loop grading its own structure; the verdict comes from BC2's JAX validation ensemble refolding
   the MPNN redesigns, which is a different model on a different sequence. Read them as two
   instruments, never as one.
2. **The rejection is JAX's, not the card's.** `traj_arm.py` leaves validation on
   `campaign_predictor`'s documented `validation="jax"` default, so these three rejections are
   BindCraft 2's own reference implementation declining a device-designed binder. That is the
   grader we would want deciding, and it is why these draws pool cleanly on verdict.

## The rule

**Acceptance is read from `2_Refolded/!_Refolded.csv`'s `outcome` column, cross-checked against
`.campaign_state.json`'s `accepted`. It is never read from a run-log stage line.** Both sources
agree on all 17 completed trajectories in the campaign after the fix; before it, neither instrument
consulted either one.
