# c14-residual-accum — residual accumulation on the pair tensor (NO-GO)

`c14-radical` ranked this #2 at a band of 0.30-0.42 s and handed it to `c14-byte-deletion`, which
closed NO-GO without taking it. Built and refuted here on its own measurements.

The band was sized correctly and is worth nothing. `ttnn.layer_norm(x, residual_input_tensor=r)`
does fuse the add away and returns 43.5 % of the site on the executed key -- but it returns only
the normed output, and a residual stream needs the sum. 0 of the 5 pair residuals in a
`PairformerLayer` can take it, proven by firing; the rewrite that keeps the sum pays the residual
read twice and measures 1.0321x (update in L1) and 1.1799x (in DRAM) against an A/A floor of
0.831 %.

    elig_screen.py      single-consumer eligibility by firing, deduped on buffer address, with the
                        LIVE-OUT rule -- without it the last residual of a PairformerLayer reads as
                        the one fusable site in the block, purely a capture-boundary artifact
    elig_qb1n0.json     413 add sites over 16 captures of the 512 aa executed path, 1 unique
                        eligible site (the diffusion transformer's closing norm, a 0.0007 s band)
    resfuse.py          the 8-arm op ladder, interleaved, benchlocked, clock forced and sampled
    resfuse_qb1n0.json  its result, with the float64 accuracy reading
    PREREG.json         written before any device arm ran; resfuse.py asserts it

Full reasoning and the verdict: `~/.coworker/state/c14-residual-accum.md`.
