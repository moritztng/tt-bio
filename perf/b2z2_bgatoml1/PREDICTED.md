# b2z2-boltzgen-atoml1-leg — written before the first device opened

Committed before any arm was read. Fixture `tests/fixtures/boltzgen/bg400.yaml` (80 designed
residues against chain A of `bgt400.cif`, 3225 target atoms), `--steps design`, BoltzGen's own
shipped protocol: `sampling_steps: 500`, `recycling_steps: 3`, `diffusion_samples: 1`. Wormhole,
whglx card 1.

**P1 — BoltzGen reaches the gate.** `TTScoreModelAdapter` subclasses `tenstorrent.DiffusionModule`,
so the atom branch of `AttentionPairBias` is on its path. The base arm's census reads `off > 0`.
The comment in `tenstorrent.py` claiming this path "exists nowhere else" is wrong, and the census
is what says so.

**P2 — the gate takes L1 on every call, and never declines.** BoltzGen's `to_keys` is built with
`W=32, H=128`, which is `ATOM_WINDOW`/`ATOM_DIM`, so the shape guard passes. The atom axis buckets
to `ceil(3225/448)*448 = 3584`, so `K = 112` windows and `live = 112 * 139264 = 15.60 MB` against
Wormhole's `0.5 * bank * 8 * 9 = 52.62 MB` budget — **30 % of budget**. Predicted census for the
L1 arm: `l1 > 0`, `dram = 0`, `shape = 0`, `off = 0`.

**P3 — the call count is a multiple of 500.** Boltz-2 reads 1200 = 200 steps x 6 atom-level calls.
If BoltzGen's atom transformer has the same depth the L1 arm reads **3000**. Any multiple of 500
confirms the gate is on the per-step path rather than a one-off; a number that is not confirms it
is somewhere else and the reading needs redoing.

**P4 — bit-exact.** The lever sets a `ttnn.MemoryConfig` and changes no operand, so the two arms
write byte-identical CIFs: equal `file_sha256` AND equal `coord_sha256`.

**P5 — the A/A control passes.** BoltzGen has no seed of its own, so `bg_arm.py` seeds torch,
numpy and random at process start. Two base arms under that seed produce the same design. If they
do not, P4 is unreadable and the verdict is INCONCLUSIVE, not GO.

**P6 — no OOM, no decline at this size, and no statement about larger targets.** 15.60 MB of a
52.62 MB budget leaves 3.4x headroom at 3225 atoms. The gate's own arithmetic puts the decline at
K = 378 (~12100 atoms on Wormhole), above BoltzGen's largest committed ladder rung of 14786 —
so at the top rung the gate is predicted to decline into `dram` rather than OOM. This pass does
not measure that rung and will not claim it.

**Falsifier.** Any of: a base arm that does not reproduce itself; a census with `shape > 0` or
`dram > 0`; two arms whose designs differ. The first makes this INCONCLUSIVE; the last is NO-GO.

---

# Addendum, written 02:32 UTC — before any bisect rung had produced a record

bg1024 crashed. The throw carries exact addresses, and they say where the cliff is. Per core:
L1 top **1,499,136 B**; at the moment of the throw the L1 buffers occupied the top **473,088 B**
(allocated at 1,026,048) and the op's static circular buffers needed **1,041,696 B** from the
bottom. 473,088 + 1,041,696 = 1,514,784, which is l1_top + **15,648 B** — an overflow of **1.04 %
of L1**. The clashing program ran on **8x8 = 64 cores**, not the 8x9 = 72 the gate divides by.

**P7 — the crossing is at K ~ 284.** If per-core residency is linear in K, the largest residency
that fits is 1,499,136 - 1,041,696 = **457,440 B**, and 457,440 / 473,088 x 294 = **284.3**. This
is an inference from two measured addresses, not a measurement, and its assumption is that the
op's CB requirement does not itself move with K.

**P8 — bg768 (K = 238) passes, bg580 (K = 182) passes.** Both sit below 284. bg768 runs
`base` and `l1` and the two write the same sha256; bg580 runs `l1` only and completes with
`l1: 3000`.

**P9 — bg1300 declines rather than crashes.** K = 364 is above the gate's own admit limit of
K = 360 (50,235,264 / 139,264), so the census reads `dram: 3000, l1: 0` and the run completes.
If it instead reads `l1` and dies, the admit limit is not where the arithmetic puts it.

**P10 — Boltz-2 is under the same cliff with ~7 % of margin.** `b2z2-atoml1-size-curve` measured
the fold clean at 1024 aa, which is **K = 266** — below 284, consistent with its GO, and close
enough that this is a warning and not a reassurance. Nothing here re-measures Boltz-2; P10 is a
statement about where its measured point sits relative to a cliff it never probed.
