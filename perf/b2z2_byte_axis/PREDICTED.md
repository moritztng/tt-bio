# PREDICTED — pre-registered before any device number exists

Committed on `wk/b2z2-byte-axis-reopened` before the lever runs, so a surprise is informative.
Numbers are BH unless they say WH. The model being priced in is the CORRECTED bytes model:
byte-proportional terms are **45.2 %** of the 36.3438 ms PairformerLayer span
(`results/REFIT.txt`), not the tile-count model CONTEXT §2 used.

## What the axis is worth at all

Deleting every delivered byte in the trunk block takes it to 1.826x, which is 1.299x on the
20.079 s cell. bfp8_b storage is a 0.531x byte factor, so the whole storage axis is bounded by:

| assumption | block | fold |
|---|---|---|
| bfp8_b everywhere on the pair track, realization 1.0 | 1.269x | **1.121x** |
| bfp8_b everywhere, realization 0.576 (`trimul_out`, measured, WH) | 1.139x | **1.066x** |
| what wave 1 actually got to run and keep (`trimul_out` alone) | 1.0342x (WH) | ~1.015x |

## The prediction

Wave 1 closed the axis with its **three largest sites never having run**: `z`, `trimul_mm` and
`triatt_qkv` all threw `static CBs grow to 1644960 B, beyond max L1 size of 1499136 B`. That is a
program-config selection artifact (a `ttnn.linear` picks a different block config for the smaller
tile), not a fact about bytes. The lever is to pin the config so the CBs fit, then measure.

1. **All three run once the config is pinned.** ~65 % confident. The failure is an L1 budget on a
   config the op chose, and the same op runs today at bf16 with a config that fits.
2. **Union byte cut 30-45 %** of the block's 5,453.0 MB, therefore **1.030-1.048x on the fold**
   at realization 0.576. Point estimate **1.038x**.
3. **Accuracy kills most of it.** `trimul_mm` and `triatt_qkv` are matmul OPERANDS, not storage:
   I predict at least one exceeds the 0.60 A kill line on the 512 aa fold. `z` is the coin flip.
   Expected survivor: **1.015-1.025x on the fold**, i.e. at or just above the 1.01x floor.

**FALSIFIER.** If the pinned-config arm runs all three sites, holds 512 aa under 0.60 A and lands
above **1.05x** on the fold, prediction 3 is wrong and the byte axis is a live 2x contributor.
If the union lands under **1.01x** on the fold or the sites still cannot run, the byte axis closes
for good under the corrected model and the whole term belongs to `b2z2-tile-arrival-latency`:
the fold is delivery-bound, not volume-bound.

## Why this is not a cheat

Storing the same values in fewer bytes removes no model work. bfp8_b IS a precision reduction, so
the line that matters is the competitor's: NVIDIA's Boltz-2 path runs bf16/fp16 activations, so a
bfp8_b ACTIVATION is below what the competitor uses and must be justified by the 512 aa structural
score, not waved through. 200 sampling steps, 3 recycles and full MSA depth stay.
