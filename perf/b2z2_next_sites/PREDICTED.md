# b2z2-step-fusion-next-sites — pre-registered, before any number from this row existed

Written while `sites_ab.py` was in flight on whglx card 8 and before its first block printed. The
base it predicts against is `b2z2-step-program-fusion`'s 41.6014 ms/step (WH), which two other rows
have since reproduced to 0.03 % and 0.09 %.

## P1 — `TT_BIO_ATOM_L1`: 1.02x to 1.06x on the step, best point 1.034x

`b2z2-step-matmul-group` priced the four atom matmuls at **2.183 ms/step** off-fold, operand and
result in L1 against the DRAM round trip the branch does today. That is an isolated screen with the
whole L1 to itself, and this campaign's own instance of the isolated-screen error is
`b2z2-step-program-fusion`'s **38 % over-prediction** of its step win. 2.183 x 0.62 = 1.35 ms/step
= **1.0336x**.

Two corrections pull in opposite directions and I am not claiming they cancel:

* **up** — this lever is wider than the four matmuls it was priced on. `s`, the key window, the kv
  projection, the padded query and the gate all become L1-resident, so the pad, the head split and
  the slice read an L1 operand too. Those are 251.5 us/layer at 56 GB/s.
* **down** — the probe's `l1_both` arm spread 4.4-29.0 %, the widest of any arm it ran, which is L1
  allocation jitter on an idle device. A real step allocates against the trimul and pair-bias
  consumers.

**Falsifier:** under 1.01x on the step refutes the lever at this size — that is the cell's own A/A
floor and a lever inside it is not measurable.

## P2 — `TT_BIO_ATOM_HEADS_UNPADDED`: REFUTED, 0.95x to 1.01x

This is the arm I expect to lose, and it is written down that way. The ceiling is real: the pad is
69.5 us/layer, the slice 23.5, and three quarters of the 158.8 us split is splitting zeros, so
about **133 us/layer = 0.80 ms/step = 1.0196x** is the most that is there. But the replacement is
two reshapes and two permutes over a 9.17 MB kv tensor, and this wave has already measured what a
permute costs at these shapes: `b2z2-step-program-fusion`'s epilogue screen found `permute+reshape`
at **308 us against a four-op chain's 72 us**, 4x slower. A reshape that splits the last dim from
256 to 32 is not a view in TILE layout either.

**Falsifier:** if this arm beats 1.01x, my pricing of a permute at this shape is wrong and the
epilogue screen does not generalise off its own shape.

## P3 — both arms bit-exact, `torch.equal`

A memory config moves bytes and not values. The head split is the same permutation of the same
channels, and its k/v channel order is not a guess: `_apply_kq_norm_atom` already slices `kv` as
`[..., :width]` = K and `[..., width:]` = V, so the model's own code states the layout.

**Falsifier:** a non-zero max abs on either arm means one of those two statements is false, and for
the split it means specifically that `nlp_create_qkv_heads` interleaves heads rather than
concatenating them.

## P4 — the union is not the product

If P1 lands and P2 is refuted, `both` should read within noise of `l1` alone, because the L1 lever
already removes most of what the pad and the split were paying for. If `both` beats `l1` by more
than the A/A floor while `heads` alone loses, the two levers interact and the interaction is the
finding.
