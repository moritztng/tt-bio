# PREDICTED — the diffusion step, before the first number of this pass

Written 2026-09-13 before any measurement on this branch. The census predicted the Pairformer
block's gap fraction at 15-25 % and measured 1.1 %, and published the miss. Same here.

## What I had already read, so it is not a prediction

Two priors were in front of me before I wrote this and I am not going to claim credit for them.
`b2z2-sampler-stall-split` records that a **replay loop** of one grabbed `Diffusion.__call__` on
whglx card 2 sums to 40.4144 ms of device kernel time against a 41.4820 ms bare synced wall, and
that `--diffusion_trace` was measured at 0.9948x on a single chip by `b2z2-diffusion-loop-attack`.
So "the replay loop is ~2.6 % gap" is prior art, not a prediction. What has never been measured is
the gap fraction of the step **as the fold runs it**, where the host is dispatching a live sampler
and not spinning a three-call loop, and nobody has put the eager and traced numbers side by side.

## The predictions

| quantity | predicted | why |
|---|---|---|
| gap fraction, **eager**, fold context | **6 - 14 %** | 1066 programs at a 20.7 us mean kernel. Eager dispatch costs single-digit us per program, so the host is close to the line rather than comfortably ahead of it, and the step should show a materially worse gap than the block's 1.1 % at 143 us/program. |
| gap fraction, **traced** | **2 - 5 %** | trace deletes host dispatch and nothing else. The residual is the device's own program-to-program latency, which the replay loop already puts near 1 us/program. |
| eager minus traced | **4 - 10 points** | this is the real prediction, and it is the one that decides Phase 2: if the difference is large the step has a dispatch prize the block does not, and if it is ~0 then `--diffusion_trace` measuring 0.9948x was telling the truth and there is no prize at all. |
| leading op code | **Matmul, 40 - 48 %** of step device kernel time | the step has no trimul, so the block's `GenericOpDeviceOperation` leader is absent; the sampler is DiT projections and feed-forwards. |
| second | **BinaryNg, 14 - 19 %** | 339 programs of AdaLN scale/shift and residual adds. |
| TRISC0 blocked on input tiles, over TRISC0's own duration | **55 - 70 %** | the block measures 65.7 % on WH with the corrected divisor and the per-op ratios in the starting table bracket the step's mix on both sides. |
| input : output stall ratio | **4 : 1 - 7 : 1** | Matmul is 11.40:1 and BinaryNg 2.43:1 on WH, and the step is dominated by those two. |
| fraction of the step that is programs doing no arithmetic | **12 - 18 %** | prior art says 132 programs, 15.3 %; restated here only so a miss is visible. |

## The wager I care about

**I expect eager and traced to differ by less than my own 4-10 point band** — closer to 2 points —
because `--diffusion_trace` already measured 0.9948x and a lever that removes host dispatch cannot
be worth nothing if host dispatch is worth 10 %. I have written the wider band above as the honest
prediction and this line as the hunch, so both can be scored.
