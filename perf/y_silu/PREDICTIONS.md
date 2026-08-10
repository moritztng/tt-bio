# y-silu — predictions, written before the device was opened

Committed and pushed before `perf/y_silu/*.py` ran a single line on card 0. qb1, card 0, ttnn 0.67.4,
Blackhole p150a. Every prediction names the outcome that makes it wrong.

## The figure under test

The ledger carries `activation="silu"` on `Transition` fc1 as **835.3 ms/fold**. That is T3's
`1204.9 − 369.6`: the fused `ttnn.linear(activation="silu")` against the identical bare matmul. It is
the **cost of the silu**, not the value of unfusing it. Unfusing moves the silu into a standalone
`ttnn.silu`, so the candidate is worth `835.3 − standalone`, and the subtrahend has never been
measured. Everything below is aimed at that subtrahend.

## The two competing mechanisms, stated before the measurement

**M1 — the 835.3 is SFPU arithmetic.** ttnn applies a fused activation in the dest register on the
fp32 accumulator, before the packer writes bf16. The silu is `x * sigmoid(x)`; sigmoid is a
transcendental and on Blackhole's SFPU that is several instructions per datum, not one. If M1 holds,
unfusing cannot delete the work, only relocate it, and it adds a full L1 read+write on top. **Under
M1 the naive unfuse is a LOSS.**

**M2 — the 835.3 is a program-config side effect.** `fp32_dest_acc_en=True` (both models) halves the
dest register file to 4 tiles. A fused activation has to live in dest alongside the accumulator, and
if ttnn's config derivation responds by shrinking `out_subblock_h * out_subblock_w`, or by refusing
`packer_l1_acc` because an activation must not be applied to a K-block partial, then most of the
835.3 is degraded matmul throughput and not SFPU work at all. **Under M2 the fix is bit-exact and
large**, and it is one explicit `program_config=`.

M1 and M2 are separable: H1 runs the same explicit program config twice, `fused_activation` set and
unset, every other field pinned identical.

## Numbered predictions

**P1 — census.** `transition_z` at 298 aa takes the 4D chunked path with `transition_h_chunk_size =
32` (`TRANSITION_H_CHUNK_SIZE_BIG`, since W=320 ≤ 384 and c=256), so H=320 gives **10 swiglu calls per
`transition_z` call**; `transition_s` is 3D with `x.shape[1]=320` under the batch-chunking threshold,
so **1 call**. Total `swiglu` invocations in a live 298 aa fold: **between 5000 and 7000**, with the
inherited "4840" recovered exactly as `10 × 484` from `transition_z` alone. **WRONG** if the
`transition_z` chunk count is not 10, or if the fold total falls outside 5000–7000.

**P2 — fc1 shapes.** The timed `ttnn.linear` runs `x_norm` `[1, 32, 320, 256]` against a `[256, 1024]`
weight, i.e. M=10240 rows, K=256, N=1024, output 10240×1024 bf16 = **20.97 MB**, L1-resident.
**WRONG** on any of M, K, N.

**P3 — H0, the naive unfuse.** `A − B` (fused minus bare+`ttnn.silu`) lands in **[−450, +150]
ms/fold**, and I expect the sign to be **negative**: under M1 the standalone silu does the same SFPU
work and pays 20.97 MB of extra read plus 20.97 MB of extra write that the fused path never pays.
**WRONG** outside that band. I am betting against the CTO's [−100, +250] on the low side.

**P4 — H0 arm C.** An in-place / output-aliased `ttnn.silu` saves at most the write, so
`C − B` is **between 0 and 60 ms/fold**, i.e. under 12.4 us/call. **WRONG** above 60 ms/fold — that
would mean the standalone silu is write-bound, not SFPU-bound, and would move the verdict toward M2.

**P5 — H1, the config comparison.** The auto-derived program config is **identical in both arms
except for `fused_activation`** — same `compute_with_storage_grid_size`, `in0_block_w`,
`out_subblock_h`, `out_subblock_w`, `per_core_M`, `per_core_N`. **WRONG** if any other field differs;
that would confirm M2 and make this a selection defect with a bit-exact fix. I put M2 at roughly 1 in
4, because `packer_l1_acc` and a fused activation are genuinely in tension when K spans more than one
block — and at K=256 with `_PAIR_PROJ_BW = 16` the whole contraction fits one block, which is exactly
the case where the tension disappears.

**P6 — H2, the measured SFPU roof.** A standalone `ttnn.silu` over `[1, 32, 320, 1024]` bf16 in L1
takes **80–220 us/call**, i.e. **387–1065 ms/fold** at ×4840. The L1→L1 copy floor for the same
41.94 MB of round-trip traffic is ~49 us/call (Z1 measured 425.4 GB/s one way on this part), so I
predict silu comes in at **1.6–4.5× the copy floor** — SFPU-bound, not bandwidth-bound. **WRONG**
outside 80–220 us/call, and **WRONG** on the mechanism if silu lands within 20 % of the copy floor.

**P7 — the placement of 835.3.** 835.3 ms/fold over 4840 calls is **172.6 us/call** for 10.49 M
elements = **60.8 Gelem/s**. I predict this sits **inside** the standalone-silu band of P6, i.e. the
fused activation is running at roughly the card's own SFPU rate for silu and there is no hidden
serialisation. **WRONG** if 172.6 us/call is more than 1.5× the measured standalone silu — that would
name a serialisation of SFPU against the MMU inside the matmul's block loop, which is a mechanism
worth reporting even without a fix.

**P8 — the net.** `net = 835.3 − standalone silu` lands in **[−450, +150] ms/fold** and I expect it
negative. **WRONG** outside. A negative net kills the candidate and that is a complete result.

**P9 — H3's ceiling.** H3 (silu folded into the `multiply_` that already reads `x_1` and `x_2`) can
win at most `835.3 − ε`, because it deletes the SFPU pass's traffic but not its arithmetic, and the
`multiply_` it hides inside is bandwidth-bound at 86.8 % of the L1 copy roof — an SFPU pass added to a
bandwidth-bound op is free only while SFPU time stays under the op's own transfer time. Given P6's
80–220 us/call against `multiply_`'s ~63 us/call (306.8 ms/fold ÷ 4840), **the silu does NOT fit
inside `multiply_`'s shadow** and H3's realistic ceiling is **`835.3 − (silu − 63) ≈ 200–650
ms/fold`**, not 835.3. **WRONG** if the measured standalone silu is under 63 us/call, which would make
H3 genuinely free and worth building.

**P10 — parity.** The naive unfuse is **NOT** bit-exact: the fused path applies silu to the fp32 dest
accumulator and packs once; the unfused path packs to bf16 first and applies silu to the rounded
value. `torch.equal` returns **False**. **WRONG** if it returns True. I predict max abs deviation
under 4e-3 at bf16's own ulp scale and PCC above 0.9999.

**P11 — the instrument.** qb1's load average is **18.58** as this is written (`uptime`, 32 cores,
`perfwar-of3-gate-spawn-deadlock`'s detached gate still live on card 2). Z1 measured the 298 aa fold
wall's A/A floor at **758.3 ms** at load 8–13 against X9's 224.0 ms at load 3.4–5.7. I predict this
session's fold-wall A/A floor is **above 500 ms**, so the fold wall cannot resolve a 200 ms effect and
the **stage/block wall is the instrument of record**. **WRONG** below 500 ms.

**P12 — the recommendation.** I predict the leg closes **DO NOT MERGE**, on a net that is negative or
inside the fold wall's own floor. **WRONG** if any arm clears +250 ms/fold and passes the envelope.
