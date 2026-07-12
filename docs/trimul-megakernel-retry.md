# Triangle-multiplication megakernel retry

## Result

No new kernel is promoted. A fresh production-size hardware A/B confirms that trimul is
controlled by data layout and memory traffic, not launch count, and that the current
channel-move decomposition remains the correct path. The earlier claim that batch-32
blocks an L1-sharded multicast feed is also rechecked against tt-metal `main` as of
2026-07-11: a newer batch-sharded factory exists, but its full-matrix-per-worker storage
contract cannot fit trimul at N=512-1024.

The useful deliverable is a reproducible 48-block benchmark and a narrower ceiling:
a future kernel must implement a streaming, partial-K producer/consumer schedule rather
than wrapping either existing matmul factory. No such design currently preserves
the measured contraction throughput while avoiding partial-output spills, so there is no
verified candidate to ship.

## Real 48-block trunk A/B

Hardware: qb2 physical card 0, one Blackhole P150 chip from a P300 board; tt-bio v0.2.5
branch base `d113620`; installed ttnn 0.68.0; bf16; real ESMFold2 checkpoint weights;
48 blocks / 96 trimul calls. Inputs are deterministic dense pair tensors. Timed regions
contain only the device-resident trunk plus an explicit synchronize; transfers and PCC
calculation are outside the timer. Each shape and path is warmed before measurement.

| N | raw channel moves | current decomposed moves | speedup | output PCC | max abs delta |
|---:|---:|---:|---:|---:|---:|
| 512 | 6.3949 s | 5.0356 s | **1.2699x** | 1.000000 | 0.0 |
| 1024 | 27.9773 s | 20.4597 s | **1.3674x** | 1.000000 | 0.0 |

`raw` uses one general permute for each channel move. `current` uses a channel-move plus
tile-local transpose for the inner-swapped operand, and two transposes for the return to
channel-last layout. This is a strict semantic A/B: projection, gating, contraction,
normalization, transition, weights, input, and accumulation order are unchanged.

The faster path issues two extra operations per pair chunk: eight additional launches per
trimul, or **768 additional launches per 48-block trunk**. It still saves 1.36 s at N=512
and 7.52 s at N=1024. Therefore a monolithic launch cannot win merely by collapsing
Python dispatches; it must remove substantially more memory traffic than the current
large contiguous/tile-local transfers.

Reproduce:

```bash
TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES=0 python3 scripts/trimul_megakernel_retry_bench.py --sizes 512 1024
```

## First-principles constraints

For one 32-channel contraction chunk, both operands are `[1,32,N,N]` bf16:

| N | one operand | A + B |
|---:|---:|---:|
| 512 | 16 MiB | 32 MiB |
| 1024 | 64 MiB | 128 MiB |

The active 11x10 grid has about 165 MiB aggregate L1, but that is not a pooled store:
shards must fit per core alongside matmul circular buffers, semaphores, and code. The
complete 128-channel pair state is 256 MiB at N=1024, so neither the full contraction
output nor all projected operands can remain in L1.

A whole-op kernel has only three possible contraction schedules:

1. **Recompute projected A/B per output tile.** This avoids storage but repeats the
   128x128 projection across output rows/columns, turning the O(N^2) prologue into
   O(N^3*D*H). It is asymptotically and practically worse.
2. **Materialize projected A/B.** This preserves O(N^2) projection work and feeds the
   existing matmul, but production operands spill to DRAM; this is the current
   decomposition with optimized channel moves.
3. **Stream K-bands from projection producers to contraction consumers.** This is the
   only radical design with possible upside. It reduces live operands to O(N*b*H), but
   every N-by-N-by-H output must accumulate across all K-bands. Keeping those partials
   needs 256 MiB at N=1024; spilling them once per band converts the saved operand traffic
   into larger O(N^3/b) output traffic. Assigning one output tile to a core preserves its
   accumulator, but then producer results require a cross-grid multicast schedule while
   contraction cores remain occupied. Existing ttnn factories do not expose that
   producer/consumer contract.

LN-in can share a read with the projection, and gate can be a projection epilogue. LN-out,
output gate, and output projection can only run after all four 32-channel contractions for
a spatial tile rendezvous. They are buildable as a second epilogue, but do not solve the
operand/partial-output storage conflict above. A single program could remove phase
barriers; it cannot make these dependencies overlap without dedicating cores to a
projection producer unless its measured throughput can satisfy every contraction consumer.

## Current tt-metal sharded-IN0 recheck

The check used an isolated source tree at tt-metal `main`
`1769bd090998f160771b4aace89c463cd28d6c01` (2026-07-11), not the older conclusions.

Two paths matter:

- `MatmulMultiCoreReuseMultiCastProgramConfig` still requires `fuse_batch=true` when A is
  sharded. Batch fusion is valid only when B has batch size 1. Trimul is
  `[1,32,N,N] x [1,32,N,N]`, so this fast 2D multicast feed remains unavailable.
- `MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig` is newer and does accept
  both operands batched. Its contract is A height-sharded in L1, B height-sharded in
  DRAM, and output height-sharded in L1, with each worker owning complete N-by-N batch
  matrices. Card 0 reports eight assigned DRAM workers, so batch 32 gives each worker
  four complete matrices. At N=512, A alone needs 2 MiB/core; at N=1024 it needs
  8 MiB/core. Both exceed the roughly 1.5 MiB/core L1 capacity before any circular
  buffer is allocated. It therefore cannot run production trimul and keeps B in DRAM.

This independently reconfirms the capacity ceiling while correcting its wording:
batch-32 is supported by a specialized sharded factory, but not for large square trimul
matrices. A partial-K custom factory would be new architecture, not a configuration knob.

## Ceiling: confirmed versus reopened

Confirmed:

- Large-N speed is memory-layout sensitive; larger aligned/tile-local movement wins even
  with more dispatches.
- Full A+B residency fails at N=1024, and the newest batch-sharded factory also fails its
  stricter per-core capacity test at N>=512.
- Standard matmul cannot consume a streaming projection band and retain all output
  accumulators across bands; replacing it means rebuilding its multicast/reuse schedule.
- The current path is accuracy-neutral: PCC 1.0 and max-abs 0.0 through all 48 blocks.

Reopened, but not yet a measured win:

- The statement “tt-metal has no batch-sharded trimul-capable primitive” is now too broad.
  A dedicated batch-sharded factory exists and proves the routing model is supported.
  A future partial-K variant could keep one projected band in L1 and stream B from DRAM.
  It must be benchmarked inside the 48-block trunk before any speedup claim.

No fabricated megakernel number is reported. The prior custom-kernel measurements were
not reused as evidence, and no speculative speedup is presented as measured. Since no
candidate beats the current path with verified PCC and full-trunk speedup, the release
gate is intentionally not changed and no performance-critical default is altered.
