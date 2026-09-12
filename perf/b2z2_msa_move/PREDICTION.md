# `b2z2-msa-movement-attack` — predictions, written before the first device run

Committed before `msa_move.py` opened a device. Branch `wk/b2z2-msa-movement-attack`, off
`wk/b2z2-msa-layer-census` @ `dfb8b9e08`. Card whglx 9, one Wormhole_B0, pinned, lease held.
Everything below is WH. The published cell is Blackhole; fractions transfer, seconds do not.

## P0 — the brief's headroom arithmetic is counted in the wrong bytes, and I predict it moves

The brief says the track moves **11.6 GB per call**, and that over 238.5 ms that is ~49 GB/s
against a movement-free ~161 GB/s. That 11.6 GB is `dram_rd 9901.1 + l1_rd 1744.8 = 11645.9 MB`
in `perf/b2z2_msa_census/split_layer_wh_c1.json` — **bytes READ, DRAM and L1 together**. The roof
it is about to be compared against (`ttnn.clone`, counted 2N/t) is **total DRAM interface
traffic**. Those are not the same currency, and the census's own table carries the other half:
`dram_wr 7998.7 MB`.

**P0: in the roof's own currency the call moves `9901.1 + 7998.7 = 17.90 GB` of DRAM, not 11.6,
so the track runs at ~75 GB/s and movement-free asks ~249 GB/s — 1.54x what the brief predicts
for both.** Falsified if the census CSV re-read gives a DRAM read+write total inside 11.6-13 GB.

## P1 — the measured Wormhole roof

`ttnn.clone` DRAM->DRAM, 128 MiB bf16, counted `2N/t`, same method as
`perf/bioir_roofline/roofs_bh.py` which gave 390.7 GB/s on a Blackhole p300c processor.

**P1: 200 GB/s on one Wormhole_B0, bracket 160-250.** A 2-read-1-write `ttnn.add` roof lands
within 12 % above it, as it did on Blackhole (429.9 against 390.7).

## P2 — the headroom, and this row's pre-registered falsifier

**P2a: the track's measured bandwidth is ~37 % of the measured roof** (75 / 200). The brief's
falsifier — *"if the track's measured bandwidth is already above 60 % of the measured roof, the
headroom this row is built on does not exist"* — is NOT tripped. The row is a GO.

**P2b: and the movement-free bound IS cut off by the roof on this track too.** 17.90 GB in
72.04 ms asks 249 GB/s of a part I predict delivers 200. So the honest multiplier is the
roof-limited one, **~2.7x, not 3.311x** — still the largest of the three blocks (the trunk's
roof-limited rung is 1.7641x on Blackhole), so the row's premise survives in rank order while
its headline number does not. If the roof measures above 249 GB/s, P2b is refuted and 3.3110x
stands unqualified.

## P3 — where the bytes are, and the lever

Post-`proj_z`-lever, `pair_weighted_averaging` moves ~5.9 GB of DRAM per call. Of that,
**1073.8 MB is the normed MSA tensor `mc` read SIXTEEN times** — eight `proj_m` and eight
`proj_g`, each streaming the whole 67.1 MB `[1024, 512, 64]` to write a 32-wide slice. That is
the same defect `proj_z` had on the pair tensor, on the other operand, and it is the largest
single repeated read left in the layer.

`_PWA_L1_NORM` already exists for exactly this and **refuses at this size**: the census capture
shows the `1x512x512x128` layer_norm writing 67.1 MB to DRAM, so 67.1 MB does not fit the
grid's L1. **The lever is to make it fit by row-blocking the head loop**, which the class
already has a path for (`pwa_depth_block` / `run(blk)`), so a block of `depth/k` rows is
L1-resident and every head reads it over the NoC instead of the DRAM interface. The cost is
`k-1` extra `m_norm` passes' worth of programs at 18.27 us each and one concat.

**P3: the best block is 2 or 4 blocks of 512/256 rows, and the layer goes 230.3 -> 215-222 ms,
a ratio of 1.04-1.07x against an A/A floor under 1.005x.** Bit-exact, because the only thing a
memory config can change about a matmul's arithmetic is `in0_block_w` and this lever does not
touch it (`_attn_value_program_config`'s measured rule, 8 shape classes x ~50 arms).

**P3-falsifier: if the best block size over a `k in {1,2,4,8}` sweep is `k=1` — today's whole
path — the residency lever is dead on this shape and I say so.** The per-program constant is
8.7 % of the wait and the blocking pays it `k-1` times over; if L1 residency does not buy more
than that, there is nothing here.

## P4 — the track after the lever

**P4: the MSA track goes 3.7054 -> 3.45-3.57 s of a ~41.3 s Wormhole fold.** Not claimable as a
fold ratio: 0.14-0.25 s is under this fixture's 1.143 % fold A/A floor, so the track delta
measured with a device sync on both sides of every call is the number this row quotes, exactly
as the census did.
