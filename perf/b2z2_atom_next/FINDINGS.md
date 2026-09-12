# The atom path, priced per line, and the one bit-exact lever left in it

whglx card 9, one Wormhole_B0 of the 32-chip mesh, ttnn 0.68.0, `cdk2x2_512.yaml` + its fixed
35-row a3m, 3 recycles, 200 sampling steps, seed 0. **Wormhole. The published cell is Blackhole;
the one BH figure here is labelled PROJECTED.**

## 1. The census

`atom_census.py` splits `b2z2-step-program-fusion`'s committed site map (1066 device programs
aligned to 1590 ttnn calls) into its blocks. Offline, no device.

| block | ms/step | programs |
|---|---|---|
| atom encoder, 3 layers | 6.324 | 106 |
| token DiT, 24 layers | 27.893 | 855 |
| atom decoder, 3 layers | 6.016 | 105 |
| **atom path** | **12.340 (30.57 %)** | **211** |

One atom layer is 2.012 ms and 36 programs, and two thirds of it is layout: key-window build
647.2 us, K/V projection 249.5, head split (pad + `nlp_create_qkv_heads` + slice) 251.5,
transition 388.0, AdaLN x2 129.3, SDPA 85.0, q/gate/out 121.9.

## 2. `TT_BIO_ATOM_KV_PREPROJ`: the projection goes between the two halves of the gather

Each atom sits in H/W = 4 key windows, so projecting the GATHERED tensor projects 17920 rows where
the atom axis has 4480. A linear is per-row, so `linear(gather(s)) == gather(linear(s))`. The
gather's two halves are a sub-tile front shift (ROW_MAJOR, the only op whose cost is set by the
channel count) and four tile-aligned block slices; the projection commutes with both, so it goes
between them: shift on 128 channels, project on the atom axis, block-slice on 256.

| chain, one atom layer, production shape | us |
|---|---|
| key window on, gather then project | 424.58 |
| project then gather | 347.17 |
| **shift, project, block-slice** | **327.65** (1.29584x) |
| K/V matmul inside it | 254.10 -> 67.13 |
| sub-tile shift | 93.52 at 128 channels, 154.51 at 256 |

Bit-exact against the shipped chain (`torch.equal`, max abs 0.0), negative control differs.
Depends on the K/V projection having no bias, so the shift's zero rows stay zero.

## 3. On the step, profiler off, interleaved

| arm | ms/step | vs base | marginal |
|---|---|---|---|
| base | 41.5745 | 1.00000x | |
| `TT_BIO_ATOM_KEY_WINDOW` | 38.6984 | 1.07432x | |
| + `TT_BIO_ATOM_KV_PREPROJ` | **38.1017** | **1.09115x** | **1.01566x** |

The key window arm reproduces `b2z2-step-program-fusion`'s 1.07444x to 0.01 % on a different chip.
Step output bit-exact between the two window arms (max abs 0.0), self-repeat exact, negative
control differs. PROJECTED on BH from the cell's 26.400 ms step: 24.195 ms, 1.0224x on the fold.

## 4. The lever changes the program count by zero

9 programs before (3 shift + 4 slice + 1 concat + 1 matmul), 9 after, and the step still falls
0.5967 ms, because the matmul it shrinks is 254 us — 26x the 9.76 us per-program constant. A
program-count model prices this lever at exactly 0. `max(kernel duration, 9.76 us)` prices it right.

## 5. Refuted: the q pad

`nlp_create_qkv_heads` requires q and kv to share a sequence length, so q is padded 32 -> 128,
split, and sliced back (250.74 us/layer). Doing the split by hand (reshape + permute per tensor,
at their own row counts) is bit-exact on q, k and v and **263.37 -> 3203.50 us, 0.0822x**.
