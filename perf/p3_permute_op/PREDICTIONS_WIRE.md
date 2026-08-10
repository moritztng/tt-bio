# p3-permute-wire — predictions, recorded before the device was opened

X6, finishing X4 (`wk/protenix-trunk--p3-permute-op` @ `92e460b9`). qb2 card 2, ttnn 0.68.0.
Committed and pushed before the first device open. Every number below is a claim I can be wrong
about; the ones I get wrong stay on the record as wrong.

## The cache

**W1 — the descriptor is mutable and the program cache honours it.** `ttnn.KernelDescriptor`'s
`common_runtime_args` is assignable from Python on 0.68.0, and `ttnn.generic_op` picks up a mutated
value on a program-cache **hit**, because the program hash covers `kernel.common_runtime_args.size()`
and not its values (read out of `generic_op_device_operation.cpp:hash_kernel` in the v0.74 source on
this host; the wheel is 0.68 and may differ). WRONG if the attribute is read-only from Python, or if
a second call with a different buffer address returns the first call's data.

**W2 — host cost after caching.** X4 measured 154.78 us of Python per call to rebuild the descriptor
at N=320. Predict the cached path costs **under 10 us** of Python per call. WRONG above 25 us.

**W3 — per-call net, host included.** At N=320 on L1, one synced call of the wired op (host + device)
beats the wheel's `ttnn.permute(x,(0,3,1,2))` at 111.38 us by **>= 1.15x**. WRONG under 1.05x.

**W4 — the cache key.** I will key on
`(N, in-buffer-type, out-buffer-type, dtype, layout, grid.x, grid.y, reader compile-time args,
writer compile-time args)`. Predict this is complete: everything else in the descriptor (CB sizes,
core ranges, the work split, and the per-core `start`/`per_core`/`Nt` runtime args) is a pure
function of `(N, grid)`, and the only per-call values left are the two buffer addresses, which live
in `common_runtime_args` and are written on every call. WRONG if two different N through one cached
path disagree with torch.

## What the lever is actually worth

**W5 — only half of T2's call count is eligible.** T2 counted **16768** in-move calls per warm 298 aa
fold. At 298 aa the trimul runs on **L1** (`TRIANGLE_MULT_L1_MAX_SEQ = 352`, H = 320), so `decompose`
is False and each chunk-loop iteration issues **one** `(0,3,1,2)` and **one** `(0,3,2,1)`. The custom
op does `(0,3,1,2)` only. Predict the counted eligible calls are **8384**, half of 16768, and the
273.8 ms/fold prize is therefore at most **~137 ms/fold** before any decomposition. WRONG if the
count comes out 16768, or if the memory config at 298 aa is DRAM.

**W6 — the other half is reachable by decomposition.** `custom(0,3,1,2)` then
`ttnn.transpose(-2,-1)` beats a single `ttnn.permute(x,(0,3,2,1))` at the fold's own shape on L1.
The decomposition is already documented bit-exact in `_transform_chunk`. WRONG if the pair loses to
the single permute.

## The fold

**W7 — the fold wall resolves the delta.** Whole-fold A/B at 298 aa, arms alternating in one process
with the model loaded once: predict a gain of **100-320 ms** on a ~32 s fold (0.3-1.0 %), and predict
the paired BASE-vs-BASE noise over the same session is **under 100 ms**, so the delta clears it.
WRONG if the noise exceeds the delta, in which case I report the trimul stage wall instead and say
the fold wall could not see it.

**W8 — clause 6, nobody else's lever moves.** `_transpose_memory_config` is called only from
`TriangleAttention`; this change touches `TriangleMultiplication`'s chunk loop and the two blocks do
not overlap in L1 lifetime. Predict the production helper returns `BufferType.L1` at the fold's own
shape **both before and after**. WRONG if it flips to DRAM.

**W9 — parity.** `torch.equal` True at the fold's own shape through the cached path, at two different
N through the same cache, and the two arms' final coordinates identical at max abs deviation 0.0.

## Q16

**W10 — the API at 0.67.4.** Predict `ttnn.generic_op`, `ttnn.ProgramDescriptor` and
`KernelDescriptor(kernel_source=...)` **do** exist on the 0.67.4 wheel qb1 runs, so a qb1 re-take is
possible and is required before merge (charter §4.8). WRONG if any of the three symbols or the
`kernel_source` keyword is missing at 0.67.4 — in which case the win ships when the wheel does.

## Not predicted, because it is settled

No prediction about a better kernel structure. P4 closed it: 64 NOC transactions per source tile is a
floor over all structures, proved by bf16 -> fp32 leaving the permute at 123.61 us while the
byte-bound clone control rose 1.36x. This pass does not go looking for one.
