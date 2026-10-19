# Trimul gate epilogue for AF2-IG (WH): design (read-only, card-free, 2026-10-19)

Worktree wk/spd-gap-bind @ 1deb04cf2. Paths: `TT` = tt_bio/tenstorrent.py, `TTL` = tt_bio/trimul_tail.py,
`K2D` = tt_bio/kernels/trimul_tail/, `KRES` = tt_bio/kernels/trimul_tail_res/, `MG` = tt_bio/mm_generic.py.
Line numbers are from this tree. The state doc's 10741 is main 28e026ba4; the same `multiply_` is TT:10648 here.

## What AF2-IG runs today (tail of `_multiply`)
- In-projection: `gin_ragged` (biased, ragged S), on by default on WH (TT:6161-6165, TT:10015-10032). It does not
  produce `g_out`, so `g_out_fused is None` at the tail.
- Tail: layer_norm -> F1 is skipped because `p_out_bias`/`g_out_bias` are not None (TT:10610-10611) ->
  `p_out = _trimul_out_proj(x, Wp, bp)` (TT:10629) and `g_out = _trimul_out_proj(x_norm_in, Wg, bg)` (TT:10644) ->
  `multiply_(p_out, g_out, SIGMOID on b)` (TT:10648-10650). Reference: af2_reference.py:288-289.
- `_trimul_out_proj` -> `_pair_proj_linear(l1_out=True, bias=...)` (TT:8931-8951): `ttnn.linear` with the bias inside the
  matmul, into **L1** (TT:6734-6737, 6762-6765). K block = whole K: `bw = k_tiles = 4` (TT:6593-6594).
- Only AF2 checkpoints have these biases: `p_out.bias` comes only from af2_weights.py:202; loaded at TT:9264-9267.

## A. What F1 computes today
- `fused_tail(xa=x_ln, xb=x_norm_in, wa=Wp, wb=Wg)` = `(xa@wa) * sigmoid(xb@wb)` in one `generic_op` (TTL:450-559).
  Operands: bf16 TILE, activations flattened to [M, K] (M = prod(padded_shape[:-1])), weights [K, N], same shape /
  dtype / memcfg pairwise (TTL:164-203). Output [.., N] bf16, DRAM or L1 (`_f1_out_mc`, TT:10653-10659).
- Two programs. (1) 2D `MG.build` transcription of the wheel's `minimal_matmul` (TTL:214-243), used whenever `_res_ok`
  fails (EPI 0, or any L1 operand/output; TTL:320-325). (2) weights-resident `KRES` (TTL:341-410), EPI >= 1 only.
  At AF2-IG's L1-resident shapes and default EPI 0 (TTL:57, `_epi` TTL:67-70) the 2D program is the one in play.
- 2D compute (K2D/compute.cpp): per output block, pass 0 / pass 1 run the K loop (one K block) into the fp32
  `intermediate_cb` (c_3) (519-587), then `copy_block` packs it to bf16 into c_4 (p) / c_5 (g) (588-594). The gate is
  `gate_block` (114-152, called at 608): copy g -> `sigmoid_bf16_tile` (bf16 branch, as `multiply_` does) -> pack to c_6
  (bf16) -> copy p and sig -> SFPU `mul_binary_tile` -> `round_bf16_tile` (ROUND = 2, integer RNE, 70-101) -> pack.
  This is the torch.equal-vs-`multiply_` order (TTL:35-44, 0/4194304 misses).
- Address plumbing: pass 1's xb/wb ride the `in2_addr` runtime slot (index 1) that minimal_matmul uses for the BIAS
  (TTL:246-254, K2D/dm_in1_sender_out.cpp:103-104). c_4/c_5/c_6 are the wheel's bias/ternary CBs, repurposed (TTL:226-231).

## B. Blocker 1: biases
Math: p = LN(x)@Wp + bp; g = x_norm_in@Wg + bg; out = p * sigmoid(g). bp enters pass 0 BEFORE the bf16 pack of p;
bg enters pass 1 BEFORE the sigmoid. The incumbent rounds: p16 = bf16(acc_p + bp), g16 = bf16(acc_g + bg) (bias added to
the fp32 accumulator inside the linear, TT:6734-6737), sig16 = bf16(sigmoid_bf16(g16)), out = RNE(p16 * sig16).
Kernel change (2D program, EPI 0 only):
1. compute.cpp:593: replace `copy_block(intermediate_cb, pass_cb, ...)` with the file's own `add_bias_block(
   intermediate_cb, pass ? bg_cb : bp_cb, pass_cb, ...)` (219-238: `add_tiles_bcast<ROW>` from the fp32 partial plus
   the bias row tile, packed to bf16) under a new `TRIMUL_TAIL_BIAS` define. That is minimal_matmul's own FUSE_BIAS
   epilogue, so the rounding order is the incumbent's: one fp32-partial + bias add, one bf16 pack, then the unchanged
   `gate_block`. Watch `add_bias_block` pushing per row (233) versus copy_block's push pattern (188).
2. Two new CBs, c_8 (bp) and c_9 (bg), N_block_tiles each (c_4..c_7 are taken, TTL:226-233).
3. Reader: dm_in1_sender_out.cpp already has the FUSE_BIAS read (111-116, 149-151, 328-339) but onto `in2_addr` and
   c_4. Point it at two new runtime slots (bp, bg addresses) and c_8/c_9. Read once per N block (the bias does not
   change with M). Bias tensors are the existing `p_out_bias`/`g_out_bias` tiles that `ttnn.linear` consumes (row 0
   of a [1, N] tile row), so ROW broadcast reads them as-is, with no `gin_ragged_bias`-style relayout (TTL:729-740).
- NOT the T1/gin_ragged method. Those seed DST with the bias before the K loop (TTL:697-701; t1bias 5fa72b0ad). That
  sums the bias inside the fp32 accumulation in a different order (and skips the srcA rounding of the FPU add), so it
  can only be equivalent, never identical.
- Bit-identity: **achievable in principle, not proven.** The bias-free F1 is torch.equal against this same tail
  (`ttnn.linear` class included, TT:8956-8966, 9001-9003), and `add_bias_block` is the wheel's bias epilogue. But the
  incumbent's bias add happens inside `ttnn.linear`'s 2D-mcast kernel, not `minimal_matmul`'s. Whether its partial CB
  format (fp32 vs bf16) and add LLK match is a device fact. The first device test settles it (G2). If it misses, the
  result is still a one-rounding-equivalent tail, graded on accuracy.

## C. Blocker 2: (4,4) block key
- What the kernel assumes: one K block (asserted for EPI >= 1, TTL:218-219; required by `eligible`, TTL:185-191).
  At (4,4) `_MM_BLOCK` is (4,4,1,4,1) against (8,8)'s (4,8,1,4,1) (TT:10829-10830). Only K_block halves, and kt = 4 is
  still one K block. Neither failure that broke (12,12) applies (out_block doubling, subblock_h halving; TTL:72-79,
  93-99). The switch already exists: `set_f1_cz128` / `TT_BIO_TRIMUL_TAIL_F1_CZ128` (TTL:111-121).
- The recorded loss (+2.019 ms/trimul on Boltz-2, TTL:101-108) predates F1_L1_OUT (TT:8970-8986). The product went to
  DRAM while the incumbent was in L1 (state/spd/of3-trimul512-passes.md:47). With L1 output, (4,4) is unmeasured.
- **Worse than described, on speed.** `MG.build` spreads N over the grid's in1 axis: padded_N = round_up(nt, gy),
  N_tiles_per_core = padded_N / gy (MG:314-321). With transpose (M > N), gy = 9 on the WH 8x9 grid, so nt = 4 gives
  1 tile per core and 4 of 9 rows working: 32 of 72 cores. Also `copy_block` and `gate_block` run the FULL
  out_block (M_block x N_block = 16 tiles, compute.cpp:593, 608) when only 4 are valid. That is 4x the SFPU gate work
  on 44 % of the grid, against a `multiply_` that uses every core. Fix: loop copy/gate over current_M x current_N at
  stride N_block (the valid tiles; output bits unchanged) and A/B `set_block` with N_block = 1 (TTL:135-151).
- Scope rule (it removes the five-model gate): admit (4,4) only when biases are passed, i.e. a new
  `eligible(..., biased=True)` key set. Boltz-2/OF3 (bias-free, (4,4)) keep declining exactly as today. TTL:87-90
  forbids a general table widening, and this is not one.

## D. Blocker 3: ragged M (mt = 195 x 224 / 32 = 1365, 1365 % 4 = 1)
- **2D program: already handled by the kernel, the clause is conservative.** The 2D transcription keeps minimal_matmul's
  ragged handling: per-core M ranges over padded_M (MG:314-320, 407), `current_M_block_tiles = min(...)` in compute
  (519-523) and in0 (dm_in0_sender.cpp:239-241). The reader stops at logical rows (matmul_dataflow_common.hpp:103-106)
  and the writer skips them (dm_in0_sender.cpp:365). Rows past M are stale CB tiles that are gated and dropped, which is
  harmless because nothing traps on Tensix. Partial blocks already happen whenever M_tiles_per_core % Mb != 0, so
  `mt % M` (TTL:199) does not even guarantee whole blocks. Action: drop it for the biased key and test parity at
  1365.
- Resident program and T2 do need it: `nb = mt // Mb` with an assert (TTL:349-363, 980-989). The 4-tile DST loop
  needs Mb*Nt % 4 == 0 (KRES/compute.cpp:101; TTL:942).
- Options there: (a) a remainder block, which needs reader clamping like gin_ragged's re-read of row S-1 (TTL:700)
  plus a writer skip. That is new indexing in three kernels. (b) Pad M up to a multiple of Mb: 1368 tiles, +0.2 %
  compute, but x would have to be re-padded or over-read. (c) **A smaller Mb that divides mt.** 1365 = 3*5*7*13, so
  Mb = 3 gives 455 blocks over 72 cores (7 vs 6 per core), with Mb*Nt = 12 % 4 = 0. Mb = 1 always divides. Mb only
  maps tiles to cores (each output tile is a whole-K contraction, TTL:135-139), so numerics are unchanged.
- **Recommend: on the 2D route, no remainder code (drop the clause). On any resident/T2 route, (c): pick the largest
  Mb <= 4 with mt % Mb == 0.**

## E. Expected saving
- Calls/fold: 4 passes (af2ig.py:157,186) x (48 Evoformer + 4 extra-MSA blocks) x 2 trimuls (af2.py:478-481,
  1215-1221) = 416, plus 4 template-stack calls once per design (af2.py:1222-1233, 1127) = **420**.
- Bytes per `multiply_`: p and g read + product written, each 1 x 195 x 224 x 128 x 2 B = 11.18 MB, so **33.5 MB/call**.
  At ~100 GB/s stock eltwise: 0.335 ms/call = 0.141 s/fold. Profiled: 0.085 s / 420 = 0.202 ms/call, about 166 GB/s.
  That is faster than the DRAM stock rate, which fits L1-resident operands (TT:6743-6765).
- What fusion removes: the `multiply_` program, the two 11.18 MB L1 packs of p/g, and one GEMM launch. What it adds:
  the gate's SFPU work (copy, sigmoid, SFPU multiply, RNE) serialized after the GEMM on the same cores. In `multiply_`
  that work is part of the 0.202 ms, so it MOVES rather than vanishes. Scaling: SFPU sigmoid is ~2.0 ms at 736 on the
  resident tail (TTL:265-272); x(43680/541696) gives ~0.16 ms/call at 195.
- Estimate: **ceiling 0.085 s/fold, expected ~0.03-0.05 s/fold**. Two checks agree. (i) 0.202 ms minus
  ~0.08-0.16 ms of relocated gate gives 0.02-0.05 s. (ii) F1's measured realisation, ~39 % of byte-priced savings
  (TT:8958-8960), gives 0.033 s. On the 2D program, with C's 32-core/4x-waste issue unfixed, it is likely a LOSS.

## F. Risk and accuracy gate
- `_multiply` is shared by every TriangleMultiplication user: AF2/AF2-IG/BC2/PXDesign scorer (af2.py), Boltz-2/BoltzGen
  (TT), ESMFold2, Protenix-v2, OpenFold3 (+template/confidence), OpenBind, OpenDDE (grep `TriangleMultiplication`).
- With the new path keyed on `p_out_bias is not None` and the biased (4,4) key, only AF2 weights reach it
  (af2_weights.py:202). Re-opened gates: AF2-IG WH + BH, and PXDesign's AF2-IG scoring. BC2's taped passes decline
  (TTL:473) and its bucket-224 untaped passes would fire, so add a BC2 design-metric check if those exist.
- Shared-source edits (compute.cpp, dm_in1_sender_out.cpp, `eligible`) must leave the bias-free build byte-identical
  (new code under `#if TRIMUL_TAIL_BIAS`, new cache-key field TTL:505-506). The cheap guard for ESMFold2/Protenix-v2
  (the (8,8) F1 users, TT:8962-8966) is an unchanged CIF digest, not a full grade.
- Gate: torch.equal against the incumbent tail if B holds (cheapest). Otherwise the AF2-IG accuracy bar used for
  T1-bias (Kabsch all-atom vs off <= 0.29 A against a 1.84 A seed floor, state/spd-gap-bind.md pass 137), WH and BH.
  Precedent warning: ragged T1-with-biases on WH measured **+0.08 s/complex** (CLOSED, state/spd-gap-bind.md:11-13).

## G. Build order and verdict
1. Host: `eligible(..., biased=True)` admits (4,4) and drops `mt % M` for that key only. Bias CBs c_8/c_9 and two
   runtime slots in `_build`. `TRIMUL_TAIL_BIAS` define. Card-free tests (mocked tensors, like tests/test_trimul_t1_bias.py):
   AF2 [1,195,224,128]@[128,128] with biases is eligible; Boltz-2/OF3 bias-free (4,4) still returns `k_tiles=4`;
   ESMFold2 (8,8) key and cache key unchanged.
2. Device micro (one card): torch.equal of `fused_tail(..., bp, bg)` against linear+bias, linear+bias, `multiply_` at
   [1,195,224,128] in L1, with the AF2 ckc, plus per-call wall time vs the three ops. GO only if it saves
   >= 0.12 ms/call (0.05 s/fold). Expect to need C's valid-tile loop and an N_block = 1 `set_block` arm first.
3. Fold A/B on WH (ABAB x s0-s2, same chip, AICLK checked), then BH and the PXDesign/BC2 checks.

**Verdict: do not build it as the route to AF2-IG WH 4x.** Its realistic yield (~0.03-0.05 s) plus the clone
(~0.036 s, 1deb04cf2) is ~0.07-0.09 s, short of the 0.12 s needed. Even the 0.085 s ceiling plus 0.036 s only just
touches 0.12 s. Two blockers are smaller than described: ragged M is an over-strict clause on the 2D route, and
(4,4) can be scoped AF2-only, so no five-model gate. But the (4,4) 2D grid mapping (32/72 cores, 4x gate waste) is a
new, worse speed problem. Bit-identity is plausible, not proven. Worth doing only as a cheap step-2 micro-bench
(about a day of host + kernel work) to replace the estimate with a number. A bigger lever would be the T2 GATE form
with biases (LN + both projections + gate + residual add in one kernel, TTL:830-836, TT:10553-10584). Price its
layer_norm/add_ share from the devprof site table before designing it.
