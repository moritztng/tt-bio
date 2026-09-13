# PREDICTED — written before the first device number exists

`b2z2-msa-layer-census`, whglx card 17, Wormhole. Committed before `msa_probe.py` ran once.

The question: `MSALayer` costs `139.47 ms + 0.0995 ms/padded row` (CONTEXT §2-CORRECTION, CONTESTED).
The 139.47 ms row-independent constant is the same *shape* as the diffusion step's 62.9 % per-program
constant. Is it the same *mechanism*?

The two known answers are mirror images. Trunk Pairformer block: 66.1 % bytes / 7.0 % per-program,
14.34 us/program over 232 compute programs of mean 143.4 us. Diffusion step: 23.9 % bytes /
62.9 % per-program, 9.76 us/program over 934 compute programs of mean 20.0 us. The discriminator
between them is program COUNT and mean program DURATION, not the block's identity.

## P1 — the MSA track's input wait is BYTE-shaped, not program-shaped

Nearer the trunk than the sampler: **bytes >= 50 %** of the fitted input wait and the
**per-program term <= 25 %**.

Two reasons, both from measurement already on the books. (a) `b2z2-msa-track-attack` measured that
45.58 ms of the 139.98 ms intercept is four pure layout passes over `OuterProductMean`'s
`[I, C*D, J]` product, 537 MB at 512 tokens — bytes with no arithmetic under them. (b) an
`MSALayer` *contains* a `PairformerNoSeq`, which is 59 % of the intercept and is exactly the block
that fits at 66.1 % bytes. A weighted guess over those two alone already lands above 50 % bytes.

**Falsifier:** the per-program term exceeds 40 % of the fitted input wait, or bytes come in under
50 %. Either way P1 is dead and the MSA track is the sampler's shape, not the trunk's.

## P2 — the layer runs 300-450 compute programs of mean duration >= 60 us

The contained `PairformerNoSeq` alone is ~230 on the trunk reader. Mean duration is the actual
discriminator: the sampler's constant dominates because it pays 9.76 us against a 20.0 us mean.
At >= 60 us mean the same constant is structurally under 20 %.

**Falsifier:** > 700 compute programs, or a mean program duration under 40 us.

## P3 — the largest lever the fit names is OuterProductMean's layout chain, not fusion of short programs

`permute` + `reshape` + two `to_layout` on the 537 MB product are 24.5 ms per call on WH
(`b2z2-msa-track-attack`, per-op census), 0.39 s/fold WH and ~0.19 s/fold projected BH, and they
compute nothing. If P1 holds they are also the single largest byte site in the layer.

Quantified: those four ops are **>= 15 %** of the `MSALayer` call, and eliding the layout is worth
**>= 1.10x on the layer unit** and **<= 1.02x on the fold** (the track is ~9.6 % of the fold, so no
MSA lever can be large at fold level; anything claiming more is measuring something else).

**Falsifier:** the four layout ops come in under 10 % of the call, or a built elision moves the layer
by less than 1.05x.

## P4 — the fixture carries 1024 padded rows against 35 real ones, and the track is 8-11 % of the fold

`MSA_PAD_MULTIPLE = 1024` (`tenstorrent.py:671`) and `cdk2x2_512.a3m` holds 35 sequences, so
`pad_amount(n_msa, 1024)` should put the row axis at 1024 with ~3.4 % of it real. 16 `MSALayer`
calls per fold (4 blocks x 4 trunk passes at 3 recycles).

**Falsifier:** the measured padded depth is not 1024, or the measured MSA stage is outside 8-11 % of
the measured fold wall.

## What is NOT being attempted

No depth lever. Full MSA depth stays: the ladder that reduces how many MSA rows the model reads was
killed at 0.709 / 0.565 A per pseudo-domain and is forbidden by rule 0 regardless. Every number here
is taken at the shipped 1024 bucket.
