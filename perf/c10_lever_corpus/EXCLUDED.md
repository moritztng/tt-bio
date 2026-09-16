# Excluded — do not re-run these

Each row names what was measured, and whether that measurement carried a clock. Where a reading was
taken at an unrecorded clock that is a fact about the evidence, not permission to re-run it: a
throttled clock cannot turn a negative ratio positive, and for the byte class it moves the ratio the
wrong way for that argument (see `clock_sensitivity.py`, section A).

| what | reading | clock | why it stays closed |
|---|---|---|---|
| bfp8 on the pair track | **0.949x** (slower), parity rejects at **1.496 A** against a 0.60 A bar | unrecorded | Slower and over the accuracy bar. The block format also disqualifies fused kernels. `_pair_proj_minimal_matmul` declines 66 of 66 calls on the `kt != 8` gate with both operands already bf16, so the dtype is never consulted; `eligible_back`'s consumer cannot run at 512 aa on Wormhole L1 at all. Dead twice over. |
| HiFi3 end to end | null | unrecorded | No effect to measure. |
| grid coverage | 103.8 of 110 cores, prize **0.000 s** | unrecorded | There is no idle-core time to recover. **But see the shortlist**: `roof-tri-close` later measured that *using* the wide grid is counterproductive on two classes, which is a different question this row never asked. |
| CB depth alone | 0.9964x and 0.9676x for `K_block` 1->2 and 1->4 | unrecorded | The producer has one block, so there is nothing to prefetch. |
| `--fast` at 512 aa | **0.95x** (slower) | unrecorded | Slower at this size. |
| tensor parallelism | trunk shard 1.3422x / 1.6404x / 1.9424x on the block at 2/4/8 chips, capped 2.600x; two-chip tensor shard 1.1364x on the trunk | unrecorded | Ruled out by Moritz on 2026-09-13 in favour of data parallelism across chips. Two chips folding independently give 2.0x throughput against the shard's 1.1364x. |
| `b_shard` | hands back **4.6 %** of the trunk, slower in 7 of 7 reps, A/A floor 0.99863x, p = 0.0078 | unrecorded | A fold regression, and tensor parallelism as well. |
| the Pairformer megakernel | priced ~1.09x, declined; the fusion population lost 2.9 % | unrecorded | Declined on its own measurement. |
| the MSA axis shard | `136.5814 ms + 0.097335 ms/row`, R² 0.996770, **57.812 % constant** against a 45 % bar | unrecorded | Measured NO-GO, and tensor parallelism. |
| the token axis shard | 56.3 % of the token DiT does not depend on token count; 1.2825x with a free link, 1.0411x with the real one | unrecorded | Measured NO-GO, and tensor parallelism. |
| byte-ranked fusion | five separate ceilings; `roof-fuse-trimul-out` deleted 268.4 MB bit-exactly and got slower; `roof-fuse-qkv-sdpa`'s ranked prize had the wrong sign (+268.4 MB) | unrecorded | The byte ledger is a lower bound on value, not a ranking of it. Price a deleted read by what its reader costs. |
| the 2.671 s roofline residual | **STOP**, no lever in it; 1.876 s is not device work | unrecorded | Nothing to take. |
| the token DiT softmax round trip | the 80.53 GB/fold round trip is already collected on main | unrecorded | Already banked. |
| `--diffusion_trace` on one chip | **0.9779x** / refuted at 0.9948x | unrecorded | Refuted on a single chip. It is 1.0404x on a mesh, which is the multi-chip route and therefore out of scope. |
| `TT_BIO_ATOM_L1` | 1.08461x WH step, bit-exact, flat in size | unrecorded | **L1 overflow at K = 294 (1152 aa)**, reproduced byte-identically on Boltz-2 and BoltzGen, at a size that folds today at shipped defaults. Crashing a size users get is a hard stop. |
| `TT_BIO_ATOM_KEY_WINDOW` | 1.02744x BH fold | unrecorded | Computes the **wrong gather**: max abs 4.21875 against a float64 reference, 13 wrong windows. `TT_BIO_ATOM_SHIFT_GATHER` reads 0.0 on the same test and ships. |
| `TT_BIO_ATOM_KV_PREPROJ` | 1.01566x WH step, on top of the defective key window | unrecorded | Marginal contribution on the cell measured zero within the floor, fold and step alike. |
| the 200 -> 50 sampling-step cut | **1.2226x, 3.586 s**, measured paired and interleaved on qb2 card 3, 5 reps per arm, spread 1.06 %. Sublinear: 4.00x fewer steps for only 3.19x less sampler time | unrecorded | **This is the single largest measured ratio in the corpus and it is a cheat.** It does less of the model's own work. Recorded here only so nobody rediscovers it and mistakes it for a lever. |
| fewer sampling steps or recycles generally | n/a | n/a | Doing less of the model's own work is a cheat, not a lever. |

Two rows that look like dead ends and are not:

* **The Transition row chunk** is not blocked by a runtime. It ships default-on for Blackhole today,
  bounded to a 128-wide pair channel. Its open item is one A/B on a quiet box, not a new build.
* **`TT_BIO_UNFUSED_SILU`** is held on a measured Protenix-v2 CA-lDDT loss of 0.0509 and 0.0721,
  because it is a shared engine default. It is not held on bit-exactness, and that distinction
  matters if it is ever re-opened per-model — which `unified-solution-not-per-model-patches` forbids.


## Three class ceilings, so a new proposal can be checked against them before it costs a chip

| class | measured ceiling | source |
|---|---|---|
| the byte axis, end to end | delete **every instrumented byte** and the fold is still 15.7 s = **1.470x**; realistic 20-30 % deletion is **1.07-1.14x** | `b2x-baseline-attrib` |
| any perfect dispatch lever | **1.050x** on the fold, **1.225x** only if every serial host byte goes too | `bioir-dispatch-graph`, `b2x-op-cost-curve` |
| what the compute thread is actually waiting for | **57.0 %** of TRISC1's resident time is blocked on CB wait-front, 9.7 % on output room, **33.3 % computing** | `b2z-kernel-cycle-census` |

All three were taken on older, larger cells at unrecorded clocks. They are class statements, not
seconds. The first one is the direct answer to "were the byte levers under-priced": played
perfectly, the whole class caps at 1.470x on the cell it was measured on, and reaching 10.0 s from a
1350 MHz fold needs 1.4275x.
