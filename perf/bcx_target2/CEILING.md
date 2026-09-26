# Which targets fit on one chip

One number decides it: **target residues + binder residues must be 352 or fewer.**

`bcx-large` measured the single-chip ceiling on Blackhole: n=352 completes, n=384 exhausts DRAM
at 34.08 of 34.226 GB. BindCraft 2 pads the complex to a multiple of 32 before it folds
(`padded_prediction_length` in `bindcraft/af2.py:52`, `-(-n // 32) * 32`), so 352 is 11 buckets
and the next residue costs a whole bucket: n=353 pads to 384 and refuses. There is no margin to
shave between 352 and 384, and no partial bucket to land in.

n is the whole complex. A 129 aa target with a 111 aa binder is n=240, which pads to 256.

## The one-minute answer on a published target

Take the target length L. The longest binder that fits is **352 - L**. Then:

| L | longest binder | what to do |
|---|---|---|
| up to 172 | 180 or more | ship `binder_lengths` as it comes, every draw fits |
| 173 to 292 | 59 to 179 | set `binder_lengths` to `[60, 352 - L]` |
| 293 and up | under 60 | one chip cannot take it |

BindCraft 2 draws a binder length per trajectory from an inclusive range, and a two-element
`binder_lengths` is that range, not two choices (`campaign_binder_lengths`,
`bindcraft/settings.py:507`). The shipped default is 60 to 180. So a target of 172 aa or less
needs no edit at all, and the only targets that need a thought are the ones between 173 and 292.

Above 292 aa the answer is no rather than "use four chips". Four chips hold 136.9 GB and would
fit n around 1396, but one gradient step spanning four chips is tensor parallelism, which is a
standing hard stop. Crop the target to an epitope instead.

Multi-target campaigns pad every target to the longest one
(`target_pad_length`, `bindcraft/campaign.py:260`), so there L is the longest target, not each.
