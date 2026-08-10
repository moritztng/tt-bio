# X3 `protenix-trunk--p3-msa-untilize` — predictions, written before the device was opened

qb1 card 2, ttnn 0.67.4. Campaign absolutes. Committed and pushed before the first `get_device()`.

## Deliverable 1 — Q14, the 8x

Two facts I can settle without the device, and I am recording them here so they cannot be
retro-fitted:

- **The brief's third candidate is dead on inspection.** The untilized tensor is
  `z = (rows*C, D*J)` with `C = D = 32` and `J` the token count, so the last dimension is
  `32 x 298 = 9536` **whatever the MSA depth is**. Depth sets `S`, the contraction length, which
  is the *inner* dimension of the producing matmul and never appears in the untilize's shape.
  So "the MSA depth reached sets the last-dim width" cannot be the explanation. Two candidates
  remain: the wheel (0.67.4 vs 0.68.0) and the host/card.
- **pc card 0 and qb1/qb2 are the same silicon.** `tt-smi -ls` on both reports Blackhole `p150a`.
  So a Wormhole-vs-Blackhole untilize-implementation split is not available as an explanation
  either. If it is the host, it is the host, not the arch.

**D1.1 — it reproduces.** `ttnn.to_layout(TILE -> ROW_MAJOR)` on `(9536, 9536)` bf16 lands at
**20000-45000 us/call** on qb1 at 0.67.4, i.e. within 40 % of P5's 35627 us. Wrong if it comes in
under 5000 us, which would put the defect on qb2's side of the wheel/host split and kill the lever
here. Confidence moderate, not high: pc is the same board type and apparently does not have it.

**D1.2 — `trunk_msa` on this card lands near qb2's 3492.9, not near pc's 1979.3**, i.e. above
3000 ms/fold. This is the clause that actually settles Q14, because it does not depend on my
attributing the gap to any single op. Wrong if it lands below 2400.

**D1.3 — the width sweep shows a cliff, not a ramp.** Untilize at fixed total bytes with the last
dim at 8 / 16 / 32 / 64 / 128 / 298 tile-columns jumps by more than 5x between two *adjacent*
points. Wrong if us/call rises smoothly with width — that would put the mechanism on
per-transaction NOC issue on the strided packer write and kill P5's circular-buffer account.

**D1.4 — the same-bytes control is fast.** `(298, 1024, 298)` untilizes in 900-1600 us,
reproducing P5's 1229.5 us within the card gap. If this one is *also* slow on qb1 the defect is not
the last dim and P5's whole account collapses.

## Deliverable 2 — the fix, if it reproduces

**D2.1** Chunking the untilize along the *last* dim into 32 blocks of `(9536, 298)` — each 10 tile
columns wide, the regime that reaches ~296 GB/s — runs the whole TILE->RM->reshape->TILE->permute
chain **at least 5x faster** than today's single wide untilize. Wrong if under 2x, which would mean
the per-op cost is dominated by something the chunking multiplies by 32.

**D2.2** The chunked arm's last-dim ordering is `(d, c)` where production is `(c, d)`. Compensating
by permuting `o_weight`'s rows on the host is **not** guaranteed bit-exact: it permutes the K index
of the consumer's reduction, and a permuted fp32 accumulation is not required to give the same
bits. I predict `torch.equal` **False** on that arm and I will not ship it on an argument. The arm
that restores the exact `(c, d)` order before the consumer is the one I predict `torch.equal`
**True**. This prediction exists because P5's A1 arm died exactly here (max abs 0.814).

**D2.3** Removing the untilize entirely is not available: `ttnn.reshape` on a TILE tensor that
changes the last dimension has to relayout, so the ROW_MAJOR trip is the relayout, not an extra.
The only escape is a reshape that does **not** change the last dim, and P5 already measured that
one as free and useless (4.4 us, untilize unchanged at 35634.7).

## Deliverable 3 — the two banked levers, on qb1

**D3.1 (C3, the PWA slice cache)** saves **15-35 ms/fold** here against P5's 26.4 on qb2, and the
`trunk_msa` stage output is `torch.equal` **True** baseline vs cached.

**D3.2 (C5, the template z hoist)** saves **10-20 ms/fold** against P5's 14.8, and
`trunk_template` stage output is `torch.equal` **True**. Order I assume for the overlap with
`protenix-trunk--p3-narrow-write`: **the hoist lands first**, so their program-config lever is
priced against 10 calls/fold, not 40. Combined ~17 ms/fold, not 24.4.

## Deliverable 4 — C4

**D4.1** The recoverable fixed term is the only thing left and it is **not worth a production
change**: batching the transitions' rows is a structural change to the MSA stack for at most the
measured fixed terms (13.92 us of the 26.38 us `layer_norm` call, 10.40 of the 37.29 us `linear`),
and I predict that prices under 30 ms/fold at the counts this stage actually runs. I expect to
close C4 with an evidenced ceiling rather than a lever.
