# of3t-trunk043ref — pre-registration

Committed before the first number of this row exists. Every branch below is named here so that
none of them can be chosen after the result is known.

## The arm

Rebuild the trunk reference from **upstream 0.4.3** — the revision `of3-p2-155k` is bound to — on
the same captured boundary `of3t_gradients/cap` and the same weights, and score the **SHIPPED**
tt-bio trunk against it. The published figures were taken against a reference built on 0.5.0:

    arm                                     z_masked     s_masked
    SHIPPED  (tt-bio transpose_bias=True)   0.279366     0.101290
    LEVER    (tt-bio transpose_bias=False)  0.049719     0.101335

`transpose_bias=True` in tt-bio is the preview2/0.4.3 orientation, so SHIPPED is the convention
correct for this checkpoint and LEVER is the port deliberately mis-set to agree with 0.5.0.

## Bars

5.0e-02 per-tensor, 2.0e-02 mass-weighted. A26: the bar an independent bf16 port can reach is
sqrt(2) x threshold = 7.0711e-02 per tensor.

## Branches, named before the run

**P1 — the pair track clears the bar (<= 5.0e-02).** The trunk's pair-track failure was a
reference-revision artefact end to end, and 5.8282 % of the model moves from no reading to
measured. Say so in the first sentence.

**P2 — the pair track clears the A26 reachable bar but not the raw bar** (5.0e-02 to 7.0711e-02).
Report both bars and say which one it clears.

**P3 — the pair track still fails (> 7.0711e-02).** The revision explains 85.6 % of the 0.279366
and something else holds the rest; name what is left and how big.

**S1 — the single track still fails against 0.4.3.** It is ours, as the two-convention invariance
already said, and nothing about the reference explains it.

**S2 — the single track clears 5.0e-02 against 0.4.3.** Then the 0.101290 was also the reference
and the campaign was wrong to call it ours. Say that plainly; it is the more interesting outcome.

**C1 — the convention-matched control does not read 0.0.** Another difference between the release
trees is live on this path and the arm is void until it is named.

## Controls, all three measured before any headline

1. **A16 zero model** — a model emitting nothing, scored against the 0.4.3 reference on this
   boundary. Measured, not assumed to be 1.
2. **A break control** — the 56 real token positions permuted in the input, weights, masks, flags
   and kernels untouched. It must move the reading, or the comparison is saturated.
3. **The convention-matched control** — 0.5.0 forced back onto 0.4.3's bias orientation must
   reproduce 0.4.3 bit for bit. Anything above the instrument floor means branch C1.

## Discipline

- A27: the reference arm's dtype policy is stated, not labelled. "float64" names a width.
- A24: the reference tree is pinned by digest, and the boundary by sha256, before loading.
- D99: both tracks, s and z, masked and padded side by side. No headline on one track.
- D95: the padding fraction beside every figure.
- A18: an agreeing forward is necessary and never sufficient.
