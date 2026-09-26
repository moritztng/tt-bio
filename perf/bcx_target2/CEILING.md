# Which targets fit on one chip

Round the binder length up to a multiple of 32. **That plus the target must be 352 or fewer.**

The drawn binder length is not what the card sees. Two paddings sit in between, and both are real:
`pad_design_chains` (`bindcraft/af2.py:56`) rounds the binder chain up to `length_bucket_size`,
32, and then `_predict_complex` (`af2.py:298`) rounds binder-plus-target up to 32 again. The
target itself is not padded in a single-target campaign, because `target_pad_length` is 0 there
(`bindcraft/campaign.py:260`) and `padded_to(0)` returns the protein unchanged.

So the token count is `ceil32(ceil32(binder) + target)`, and since 352 is itself a multiple of 32
the rule collapses to `ceil32(binder) + target <= 352`.

`bcx-large` measured the ceiling on one Blackhole: 352 tokens completes, 384 exhausts DRAM at
34.08 of 34.226 GB. `bcx-armtree` ran BindCraft 2 itself at 320 tokens for six consecutive
gradient steps with 0.0000 GB of resident rise, so 320 is the largest size real BindCraft 2 has
been measured at and 352 is the largest a gradient step of that shape has been.

## The one-minute answer on a published target

Take the target length L. The longest binder that fits is **32 x floor((352 - L) / 32)**.

| L | longest binder | what to do |
|---|---|---|
| up to 160 | 192 | ship `binder_lengths` as it comes, every draw to 180 fits |
| 161 to 192 | 160 | set `binder_lengths` to `[60, 160]` |
| 193 to 224 | 128 | `[60, 128]` |
| 225 to 256 | 96 | `[60, 96]` |
| 257 to 288 | 64 | `[60, 64]` |
| 289 and up | under 60 | one chip cannot take it, crop to an epitope |

The answer is always a multiple of 32 because the binder is bucketed before it is added. Every row
is produced by `perf/bcx_target2/padmap.py`, which calls BindCraft 2s own two padding functions on
real `Protein` objects rather than reimplementing them.

**The cutoff is 160, not 172.** A 172 aa target with the shipped 60-180 default draws 180 one time
in six: `ceil32(180)` is 192, plus 172 is 364, which rounds to 384 and is the bucket that exhausts
DRAM. Arithmetic that adds the raw lengths puts 172 + 180 = 352 safely inside and is wrong by a
whole bucket. Above 288 aa even a 60 aa binder misses, because `ceil32(60)` is 64 and 64 + 289 is
353.

Above 288 the answer is no rather than "use four chips". Four chips hold 136.9 GB and would fit
around 1396 tokens, but one gradient step spanning four chips is tensor parallelism, a standing
hard stop. Crop the target to an epitope instead.

Multi-target campaigns are the one case where the target is padded too: `target_pad_length` is set
to the longest target rounded up (`campaign.py:260`), so there L is `ceil32(longest target)` for
every target in the campaign, not each targets own length.

## Worked, on the target this row ran

HEWL is 129 aa. `32 x floor((352 - 129) / 32)` = `32 x 6` = 192, above the 180 the default range
can draw, so `examples`-style settings need no edit at all. The trajectory drew 111, which rounds
to 128, plus 129 is 257, which rounds to **288 tokens**.

    JAX_PLATFORMS=cpu PYTHONPATH=/home/ttuser/bcx_e2e/bc2 \
      python3 perf/bcx_target2/padmap.py --settings perf/bcx_target2/lysozyme.json

    target 129 aa, draws 60..180, ceiling 352
       224 tokens  <- draws 60..64 (5)
       256 tokens  <- draws 65..96 (32)
       288 tokens  <- draws 97..128 (32)
       320 tokens  <- draws 129..160 (32)
       352 tokens  <- draws 161..180 (20)
    longest draw that fits: 192

Five distinct sizes across a 121-value draw range, which is also why a campaign on one target
compiles five shapes and not 121.
