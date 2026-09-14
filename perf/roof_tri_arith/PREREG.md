# roof-triangle-arith-efficiency — registered before the first device command

## 0. The brief's premise does not survive its own source, and this is checked on paper first

The brief says the triangle rows are at "22.8 % / 12.2 % / 3.3 % / 14.2 % / 12.3 % / 3.3 % of cube"
and that the two out-projections at 3.3 % are "the worst arithmetic efficiency measured anywhere in
this fold". Those six numbers are reproduced exactly by `perf/roof_shape/weighted_pc_bh.json` as
each class's share of the 8.7955 s of covered floor SECONDS, not as its fraction of the dense cube:

    s_per_fold / sum(s_per_fold):  22.8  14.2  12.3  12.2  3.3  3.3   -> 68.0 % together
    pct_of_cube for the same rows: 14.9  15.0  13.8  14.8  13.0 13.0

So the six triangle classes do not span 3.3-22.8 % of anything. They span **13.0-15.0 % of cube**,
a 2.0-point band, and the out-projections are the worst by 0.8 points over the triangle product,
not by 4x. Registered as a premise correction before any measurement, because it changes the
question from "why is one class at 3.3 %" to "why are six unlike shapes all pinned at ~14 %".

## 1. What the existing data already refuses

`roof-shape-honest-roofs` closes with "the rest is the MAC array failing to fill at a 4-tile K".
Its own table refuses that: the triangle product is a K = 16-tile matmul (batch 128, 512x512x512)
and lands at 13.8 %, below the K = 4-tile in-projection's 15.0 %. And the same six rows span
18.1 % to 60.5 % of the measured DRAM roof while their fraction of cube moves by 2 points, so DRAM
is not the common cause either. Some third thing pins the plateau.

## 2. PREDICTED — the fidelity ladder is the discriminator

Every arm is measured at LoFi, HiFi2 and HiFi4 against a cube measured at the same three fidelities
in the same session. HiFi4 is 4 MAC passes, HiFi2 is 2, LoFi is 1. If a shape's time is set by the
MAC array, LoFi/HiFi4 approaches 4x; if it is set by anything else, it approaches 1x.

    arm                     predicted LoFi/HiFi4 speedup     what it would mean
    cube4096                3.2 - 4.0x                       control: MAC-limited, as designed
    pair_out128 (K=4,N=4)   1.00 - 1.25x                     bandwidth/pack-limited, no arith lever
    trimul_in  (K=4,N=20)   1.4 - 2.2x                       partly MAC, partly not
    triatt_in  (K=4,N=17)   1.4 - 2.2x
    trimul_einsum (K=16)    1.6 - 2.5x
    SDPA q256k256           1.3 - 2.0x                       fused softmax caps it

Centre call: **the plateau is NOT the MAC array.** I expect the four pair-tensor projections to
come in under 2.0x where the cube comes in over 3.2x, which would say the "arithmetic floor" that
`roof-shape-honest-roofs` made the campaign's headline is, at these shapes, mostly not arithmetic.
Registered outcome: if the projections show < 2.0x and the cube > 3.2x, the campaign's 11.134 s is
still the right measured floor but its NAME is wrong and the lever class it points at is wrong.

Falsifier: if every triangle arm scales 3.2-4.0x like the cube, the MAC array IS the limit, HiFi2
is worth ~2x on 68 % of the floor, and the only question left is accuracy.

## 3. Secondary ladders, same session

- K at fixed M and N: 262144 x K @ K x 128 for K = 128, 256, 512. Tests the 4-tile-K claim directly.
- N at fixed M and K: 262144 x 128 @ 128 x N for N = 128, 256, 640. Tests in0 reuse.
- core grid full vs half on one projection. Tests whether the ceiling is per-core.
- the same starved 8192^2 add the campaign uses for the DRAM roof, in this session.

## 4. Kill criteria, unchanged from the brief

Under 1.05x on the op across two interleaved sessions with an own-session A/A floor is dead. A
named mechanism with no lever is a full pass. No arm may do less of the model's own work; a
fidelity change is a precision change and is governed by the 0.60 A structure bar with the 1.84 A
seed floor beside it, never by bit-exactness.
