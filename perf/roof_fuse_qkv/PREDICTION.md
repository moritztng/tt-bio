# ROOF Phase A, qkv -> SDPA: what I expect before the device runs

Written before any measurement on this branch. `perf/roof_fuse_qkv/bytes.py` is host-only and ran
first; the device numbers below are the ones still open.

## The byte ledger says the Q arm has the wrong sign

Shapes off the same committed trace FUSION_PAIRS.md ranked
(`perf/b2z2_byte_floor/out/trace_512_wh_c10.json.gz`, row 64 and row 70): one triangle attention at
512 aa is `x = 512x512x128` (67.1 MB) projected by a `128x384` weight into q, k, v each
`512x4x512x32` (67.1 MB). So B=512, H=4, S=512, head_dim=32, C=128, two attentions per block.

The ranked prize counts the six intermediates' write + read: 805.3 MB. That count is right. It is
not the fold's byte delta, because a fused consumer does not stop reading -- it reads the producer's
operand instead. Here the producer contracts 128 channels into 32 per head, and the SDPA is split
**one head per core**, so each of the H = 4 heads re-reads all 128 channels of x.

Per attention, with q_pf q chunks per (batch, head):

| arm | DRAM bytes | delta |
|---|---|---|
| today | 67.1 (mm reads x) + 201.3 (mm writes qkv) + 67.1 (sdpa reads q) + 134.2*q_pf (sdpa reads k,v) | |
| fold Q only | 67.1 + 134.2 + **268.4** (sdpa reads x, H times) + 134.2*q_pf | **+134.2 MB** at every q_pf |
| fold q+k+v | **281.5*q_pf** (sdpa reads x once per head and q chunk) | -201.3 MB at q_pf=1, +201.3 at q_pf=4 |
| fold q+k+v, x multicast across the H head-cores | 70.4*q_pf | -402.7 MB at q_pf=1 |

**PREDICTED (Q arm): +134.2 MB per attention, +268.4 MB per block. The ranked -268.4 MB prize for Q
is the same number with the opposite sign.** It is not an implementation detail: no reader schedule
can produce a 32-channel head without reading the 128 channels it contracts.

At 512 aa the shipped ladder takes the widest q_chunk that fits, which spans the sequence, so
q_pf = 1 and the q+k+v arm is the only one on the positive side.

## PREDICTED, device

1. **Byte slope of the fused SDPA at 512 aa.** Sweeping q_chunk over 512/256/128/64 moves k and v
   from 1 to 8 re-reads with identical per-row arithmetic. Measured Blackhole p150a DRAM roof is
   435.2 GB/s = 2.30 us/MB. `k10-p1-trimul-critpath` measured this program 22.5 % input-stalled, and
   a stall fraction is not a recoverable fraction, so I predict the slope lands **below** the roof
   rate, in **0.2 - 1.0 us/MB**, and nearer the bottom of that range than the top.
2. **Q arm, op ratio.** The SDPA's own reads go 67.1 -> 268.4 MB, +201.3 MB. At the predicted slope
   that is **+40 to +200 us** on a program whose mean is 1323.8 us: **0.86x - 0.97x, a slowdown**.
   The qkv matmul gives back one 67.1 MB write, worth 0 - 67 us on a different program, which does
   not cover it.
3. **Q arm, fold ratio.** Two triangle attentions per Pairformer block: predicted **0.97x - 1.00x**,
   i.e. at best nothing, at worst a measurable loss.
4. **Full q+k+v arm, at q_pf = 1.** -201.3 MB per attention, -402.7 MB per block = 5.43 % of the
   block's 7416.9 MB. At the predicted slope that is **-40 to -200 us per attention**, but it
   recomputes the k and v projections on every core and adds a matmul prologue to a compute kernel
   whose L1 is already spent on the persistent mask CB.

## What stops me

The Q arm is killed if the byte delta is positive, which it already is by exact count. The measured
slope decides only how much it loses by. If the slope came out at or above the roof rate, the
instrument would be wrong and I would re-derive it before quoting anything.
