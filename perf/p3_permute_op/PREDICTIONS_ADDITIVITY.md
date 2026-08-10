# Z1 p3-additivity — predictions, committed before the device is opened

Q23: is X9's **+328.6 ms/fold** (`reblock_permute`, qb1 card 0, ttnn 0.67.4, pre-merge baseline)
still there on top of **`cc39a867d`**, the commit **Moritz merged** at 08:35 on 2026-08-10 carrying
X2's narrow-write, X7's L1 output and X10's two norm sites?

Host qb1 (`tt-quietbox`), ttnn 0.67.4 from `/home/ttuser/tt-bio-dev/env`, Blackhole p150a, 13x10
grid. Branch `wk/protenix-trunk--p3-additivity` = X9's ten commits cherry-picked onto `cc39a867d`.
Every prediction below names the outcome that makes it WRONG.

---

**P1 — the rebase is clean and the kernel still compiles.** X9's stack `7489e789..9ccfa6f9` replays
onto `cc39a867d` with no conflict, because X9 touches `tt_bio/tenstorrent.py` only at
`_channel_move` (a new free function) and at two lines inside `_transform_chunk`, while the merge
touches the projection helpers and `_l1_*`. The three `.cpp` kernels JIT-compile against 0.67.4 on
the merged tree. WRONG if any file conflicts or the JIT fails.

**P2 — the kernel serves 4352 of 4352 eligible calls in a live 298 aa fold**, split 4192 on the
c_z=256 pair track (4 chunks of 64 channels x 2 trimuls x 524 executions) and 160 in the template
stack (1 chunk x 2 x 80). `_trimul_chunk_size(298, 128)` reads **64** on this grid, and
`COMPUTE_GRID_MAIN` reads **(13, 10)** after the device is open. WRONG if the served count is not
4352, and CATASTROPHIC if it is 0 — that was X9's first A/B and it was an A/A.

**P3 — Q23 verdict: ADDITIVE, and I predict +230 to +400 ms/fold**, centred near X9's 328.6.
The mechanism argument for additivity: X9's kernel replaces `ttnn.permute(0,3,1,2)` on the trimul's
per-chunk `[1,298,298,64]` tensor inside `_transform_chunk`, whose memory config is
`_triangle_mul_memory_config(H)`; X7's merged L1 output is on `_trimul_out_proj` / the pair-track
projections, which run at the trimul's **tail**, after the chunk loop, and X10's merged norm sources
are in `AttentionPairBias`/PWA/the template embedder, not in the chunk loop at all. Three consumers
of the same bank budget, but not simultaneously at the chunk-loop instant. WRONG if the measured
delta is under +200 ms/fold or over +450.

**P4 — the two arms' L1 headroom at the chunk-loop instant is not tight.** Main's own comment prices
the two live 48.82 MB pair-track L1 tensors at **750.9 kB of each bank's 1427.5 kB**. The kernel's
static circular buffers are `IN_CB(2) + OUT_CB(64) + STAGE_CB(2) = 68` bf16 tiles = **139.3 kB per
core**, and its L1 output tensor is 11.37 MB over 130 banks = **87.5 kB per bank**. 750.9 + 139.3 +
87.5 = 977.7 kB, which leaves ~450 kB. I predict **free L1 per bank stays above 300 kB with the
kernel ON**, measured from the allocator on the open device rather than arithmetic. WRONG below 150 kB.

**P5 — `_L1_OUT_REFUSED` is EMPTY after live 298 aa folds with the kernel ON.** X7's merged
593.3 ms/fold is not silently part-cancelled by the flag. WRONG on any entry. This outranks the
ms/fold: it is a statement about code already on main.

**P6 — `_transpose_memory_config` still returns `BufferType.L1`** at the fold's own pair shape
`[298, 320, 256]`, read on the rebound 13x10 grid, before AND after a fold with the kernel ON. Its
2.5x-headroom fit test sees 122.06 MB of demand against 199.22 MB of capacity and takes no live
occupancy as input, so no residency change can flip it. WRONG if either read returns DRAM. This is
the 1010.9 ms/fold lever nobody in this org owns.

**P7 — no trimul circular-buffer throw at C=64 with X10's template norm resident.** Main's own
`_TEMPLATE_L1_NORM` comment records exactly this failure ("statically allocated circular buffers in
program 173 clash with L1 buffers") from a naive L1 norm and fixed it by narrowing the residency
window. Adding 139.3 kB/core of generic_op CBs is new pressure on that fix. WRONG on any throw,
including one swallowed by a try/except — a caught refusal is still zero win.

**P8 — the A/A floor on this host this pass is 150-350 ms**, bracketing X9's 224.0 ms, because three
`perfwar-*` workers rotate cards here and the load average is 4-7 right now. WRONG outside that band.
I predict the load average is **above 3.0 for the majority of timed folds** and I will log it per run.

**P9 — the two instruments keep X9's ratio to within 1.15x of X9's 1.43x**, i.e. the trimul block
wall lands at +160 to +290 ms/fold and the fold wall is 1.25-1.65x it. A ratio outside that is a
finding about additivity, not noise: the block wall replays the trimul in isolation, so if the merge
changed what the *rest* of the fold holds in L1, the two instruments move apart.

**P10 — the gate window re-derived on 13x10 at C=64 is WIDER on DRAM and NARROWER on L1 than the
shipped `(DRAM and N>=256) or (L1 and 288<=N<=352)`.** X9 measured 100 of 130 cores engaged at
Nt=10, and 130 of 130 with 14 cores double-loaded at Nt=12 (N=384), where the ratio falls under 1.0x.
So I predict the L1 win survives at N=288-352 and dies by N=384, and that the DRAM leg wins from
N=256 up with a much larger margin (X9: 2.28x at C=64 against 1.48x on L1). WRONG if the production
shape N=298 on L1 is outside my measured window — a gate that switches the win off at the production
shape is the failure mode this org has already seen.

**P11 — parity re-taken here reproduces plDDT 0.859489** to six decimals on every arm, and
`torch.equal` is True for the kernel against `ttnn.permute` at `[1,298,298,64]` on L1 and on DRAM.
The merged L1 writers are new code in BOTH arms, so they cannot move the A/B's parity, but they are
new code under this kernel and the check is re-taken rather than inherited. WRONG on any mismatch.

**P12 — the merge recommendation stays HOLD.** Even if the number is fully additive, the blast radius
X9 named is unchanged: `_channel_move` sits in the shared `TriangleMultiplication` and is reached by
five other models. The wheel is not the variable and is not re-opened (X9: 1.491x at 0.67.4 against
1.493x at 0.68.0). WRONG if the evidence lands somewhere else, which is a real possibility if Q23
comes back additive and the interaction checks all come back clean.
