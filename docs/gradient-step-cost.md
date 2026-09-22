# What a pair-track gradient step costs

`tt_bio.autograd` tapes a forward and runs the backward on device, so a pairformer stack can be
differentiated rather than only folded. This page prices one gradient step through that stack, so
you can size a training or design loop before you build it.

Everything below was measured on one Blackhole chip (`tt-quietbox2`, P300c board, PCI subsystem
`0x0046`, `TT_VISIBLE_DEVICES=0`), bf16, at AlphaFold2's pair dimensions: `c_z = 128`, 4 heads,
32 per head, SDPA chunk 128. 22 points, 30 gradient steps each, mean over steps 6 to 30. Eighteen
completed; the four that did not are 256-token runs at 6 and 8 blocks that ran out of device
memory. Four blocks at 256 tokens fits and six does not, and checkpointing does not move that
line, so whatever binds there is not the live tape. The AI clock was sampled once a second
during the runs: 223 timestamped samples, every one between 1337 and 1350 MHz and 91 % of them
at 1350, so none of these numbers is a throttled-clock artifact. Raw data:
[`perf/hallgrad/p2_floor_screen.json`](../perf/hallgrad/p2_floor_screen.json).

## Depth is the whole cost

A step over `K` sequential pairformer blocks costs `a + b*K` seconds. The fit is clean:

| tokens | a | b | worst residual | r² | K |
|---|---|---|---|---|---|
| 128 | 0.0077 s | 0.0649 s | 0.6 % | 0.999987 | 1, 2, 4, 8 |
| 256 | 0.0397 s | 0.2320 s | 0.18 % | 0.999996 | 1, 2, 4 |

`a` is not host overhead. It is everything that happens once per step rather than once per block:
the pair init, the distogram head, the host loss and the host optimizer update. At 256 tokens it
is 0.040 s against a 12.11 s step over an AF2-depth stack of 52 blocks, which is 0.33 %.

So there is no fixed cost worth amortising, and a lever is worth roughly what it is worth on one
block times the depth. Price your loop as `blocks * steps * b` and you will be within a percent.

The backward is 1.72x the forward per block at 256 tokens: 0.0854 s forward against 0.1465 s
backward, both fitted on the same points.

Gradient checkpointing is a memory lever and costs speed. Recomputing the block inside its own
backward takes `b` from 0.2320 s to 0.3184 s at 256 tokens, a 1.37x tax, and collapses the live
tape from 529 nodes to 21 at K=4. Reach for it when you are out of memory, not when you are slow.

## The per-block cost is device compute

One block's forward at these dims is `1536*N³ + 722944*N²` FLOPs plus `24*N²*c_z²` for the
transition, so a compute-bound `b` has to grow by 4.60x going from 128 to 256 tokens.

| block | b(128) | b(256) | measured | from FLOPs | share |
|---|---|---|---|---|---|
| with transition | 0.0649 s | 0.2320 s | 3.57x | 4.60x | 78 % |
| without transition | 0.0566 s | 0.2111 s | 3.73x | 4.86x | 77 % |

At 78 % of the FLOPs ratio the cost tracks the work. A stack bound by per-op dispatch would sit
far lower, because dispatch does not care how big the tensors are.

The same two points put a number on the part that does not scale. Split `b` into a term that
grows with the FLOPs and one that does not, and the shape-invariant term is 0.0185 s per block:
28 % of `b` at 128 tokens, 8 % at 256. That is the ceiling on anything that attacks launch and
dispatch, trace capture included. It buys at most 1.40x at 128 tokens and 1.09x at 256.

## It runs at 7 % of a matmul roof

The roof here is not a datasheet peak. It is `ttnn.matmul` in bf16, in the same process, on the
same chip, at the same clock, on the shape the block's linear layers actually use, 3 warm-up
calls and 20 timed with an explicit device sync:

| tokens | roof shape | roof | block achieved | fraction |
|---|---|---|---|---|
| 128 | [16384,128] @ [128,128] | 15.05 TFLOP/s | 0.99 TFLOP/s | 6.6 % |
| 256 | [65536,128] @ [128,128] | 18.65 TFLOP/s | 1.28 TFLOP/s | 6.9 % |

Achieved counts the backward as twice the forward's FLOPs, the usual convention.

Both halves of this matter and they say different things. The work is real, so the cost will not
fall to a dispatch fix. And we are doing it at a fourteenth of the rate the same chip reaches on
a dense matmul, so it is not a silicon limit either. The gap lives between the two: the pair
block is a mixed chain of triangle multiplication, triangle attention, layer norms, gating and
softmax, each op reading and writing DRAM on its own.

## What this means if you are pricing a design loop

Gradient-based design optimizes an input through a folding trunk, so its cost is the step above
times the number of optimizer rounds, plus any forward-only recycles. At AF2 depth and 256
tokens that is 12.11 s for a forward-and-backward step and 4.45 s for a forward-only pass, and a
few hundred rounds per trajectory puts a single trajectory in the tens of minutes on one chip.
The same arithmetic holds for any method with this shape, not just one.

Two things follow for anyone deciding where to spend effort. Nothing is recoverable from the
fixed cost or from dispatch, and the numbers above bound both. The headroom is entirely in the
achieved rate per block, which means fusing the block's chain so its intermediates stay in L1
instead of round-tripping through DRAM, on the backward as much as the forward.

## Reproduce

```
TT_VISIBLE_DEVICES=0 python3 perf/hallgrad/blocksweep.py \
  --out floor.json --steps 30 --ns 128,256 --ks 1,2,4,8 --transitions on,off
```

One process, one device open, one clock thread, every point measured inside it. Add
`--checkpoint` for the recompute arm. The sweep refuses a point whose backward reaches a
degenerate tape and records each point's clock window, so a result it accepts is attributable to
the window it was timed in.
