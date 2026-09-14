# roof-redteam: pre-registered predictions

Written before any computation, after reading only `RECONCILIATION.md` (wk/roof-orchestrator,
495634a45). Each line is what I expect to find, so that a later agreement is evidence and not a
result of me fitting the answer.

## Attack 1 — tile padding on the 206.706 TFLOP count

PREDICTED: the padding factor is small. 512 tokens = 16 exact tiles, 7168 atoms = 224 exact tiles,
pair channel 128 = 4 tiles, single 384 = 12 tiles. The candidates that actually pad are the MSA
depth (35 sequences -> 64, 1.83x on that phase only) and any head dimension below 32. I expect a
whole-fold padding factor in **1.05x-1.40x**, nowhere near the 4.22x needed to rescue 873 TFLOP.
The correction survives; the compute floor moves from 2.405 s to at most ~3.4 s, still not binding
against 17.34 s.

## Attack 2 — the self-declared atom-transformer undercount

PREDICTED: the 5.496 TFLOP atom-transformer row is not all attention; the MLP/projection part
does not scale with the key window. If the attention part is roughly half the row and the true
key window is 128 against a 32-wide query window (4x), the correction adds **under 20 TFLOP**,
i.e. under 10 % of 206.706. Compute floor goes to ~2.6 s. Survives.

## Attack 3 — does the cell really run 3 recycles / 200 steps

PREDICTED: confirms. Boltz-2's defaults are recycling_steps=3 and sampling_steps=200 and the
published cell has no reason to differ. Risk I am watching for: the site JSON may not record the
config at all, in which case the claim rests on the code default and I must say so rather than
assert it. If the cell were 10 recycles the fold FLOPs would rise ~2.4x and the comparison moves.

## Attack 4 — the 2.47x residual under 872.95 TFLOP

PREDICTED: I can name it, and the leading candidate is a **counting-convention factor of 2**
(MAC vs FLOP, i.e. M*N*K vs 2*M*N*K) times ~1.24 of padding/extra ops. Second candidate: 872.95
is not Pairformer-scoped at all but a whole-trunk figure that includes the MSA module, and the
two documents that call it "the Pairformer's" are repeating a label loosely. If the second is
true, the orchestrator's headline claim is wrong in its most important sentence, though the
verdict (compute does not bind) can still survive on the 206.706 side alone.

## Attack 5 — 429.9 vs 444.9 GB/s

PREDICTED: they measure different things only in rig and shape, not in kind; both are
read+write DRAM streams on Blackhole. The gap is 3.4 %, inside p300c clock-boost variance
(memory: p300c boosts 800->1350 MHz and skews isolated micro-benchmarks). For a fold-level floor
the roof must be counted the same way the traffic is counted: the 3.405 TB figure sums reads and
writes, so the roof must be a summed read+write roof. I expect 429.9 to be the correct one to
quote for the p300c cell and the choice to change no verdict (7.920 s vs 7.653 s).

## Attack 6 — fusion_pairs.py classification

PREDICTED: the EPILOGUE call on `generic_minimal_matmul -> multiply_` is the weakest one. A fused
matmul's pack stage writes DST to the output CB as tiles complete; a following elementwise
multiply by a tensor that is not already an operand of that matmul cannot in general be folded
into the pack without the second operand being resident. I expect a **partial break**: some of
the 967.8 MB epilogue bucket is real (bias/activation shapes) and some requires a second full
tensor to be co-resident, which is not a free epilogue. I predict I can move the epilogue number
by more than 10 %.
