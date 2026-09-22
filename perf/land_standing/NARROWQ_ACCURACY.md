# narrow-q accuracy: bit-exact at its cell, and by construction rather than by luck

`TT_BIO_TRIATT_NARROW_Q_FALLBACK` on rf3 at 896 aa, the one rung where the reach audit found the
flag actually changes what a production fold does.

## Measured

**One distinct CIF sha256 across all 12 legs of rf3 at 896 aa**, spanning two independent
processes (`narrowq_rf3_retake1s_work`, 9 legs, 2026-09-22 22:05Z; `narrowq_bank/work_20260922T230350Z_on_c0`,
3 legs, 23:03Z) and both arms:

    612a6e1ad3fdba1a5dec2d7c7b572b5d   all 12

So **0.000000 A** against the **0.60 A** default bar, with the **1.84 A** seed floor beside it.

**The lever demonstrably fired in those same legs**, which is what stops this being a vacuous
digest match: the two arms differ by **+9.50 s (1.1005x)** on the pooled cell. A flag that never
took effect would have produced the identical digest AND the identical runtime.

## Why it is bit-exact structurally, not coincidentally

`_tri_att_q_chunks`' own docstring, on the wide-q sibling that shares the mechanism: *"q_chunk
only splits output rows and the online softmax reduces over k, so no reduction order changes"*,
recorded as `torch.equal` to the shipped config at every size it was swept over. The narrow
fallback moves the same knob in the same direction, so the measured digest match is the expected
result rather than a surprise worth arguing about.

## Reach, and the limit of this evidence

The flag only changes behaviour where L1 refuses every wider dividing chunk AND the production
pick does not divide the padded length. A length whose fallback already divides -- 768, 1024,
every multiple of 256 -- returns the identical tuple with the flag either way, so those are
no-ops **by construction**, which is also why the 768 aa negative control cannot be informative
about the lever.

Auditing all 142 baseline censuses for `fill_preconditions` declines leaves **rf3 at 896 and
1088** as the only dark-and-firing cells; nesso1, openfold3 and opendde are dark at rungs where
the policy is inert, so they are dark for their own reasons (`narrowq_reach_audit.py`).

**Not established here: 1088 aa.** It fires and refuses more configs, so it stresses the L1 term
hardest and is the rung that could still falsify a default flip. Bit-exactness is expected to
carry (the mechanism is size-independent); the *timing* sign is not, because a narrower chunk
re-reads K and V once more per chunk and the win at 896 comes from keeping the fused path rather
than from the chunk width itself.
