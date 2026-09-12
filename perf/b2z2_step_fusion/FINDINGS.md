# The diffusion step, priced site by site, and the first lever off that price

whglx card 2, one Wormhole_B0 of the 32-chip mesh, 8x9 grid, ttnn 0.68.0, `cdk2x2_512.yaml` +
its fixed 35-row a3m, 3 recycles, 200 sampling steps, seed 0. **Wormhole. The published cell is
Blackhole; every BH figure here is labelled PROJECTED.**

## 1. The map: 1066 device programs, aligned to the 1590 ttnn calls that issue them

`b2z2-sampler-stall-split` measured the step's input wait at 55.8 % of math-thread residency and
found **62.9 % of that wait is a per-program constant of 9.76 us**, not bytes. The currency on
this block is therefore programs, and a per-op-code histogram cannot say which LINE of
`tenstorrent.py` owns them. `site_cost.py` gets that by aligning two orderings of the same call:
the graph capture's top-level ttnn sequence and the armed capture's device sequence. The walk
consumes **1066 of 1066** and reproduces the committed per-op-code census (Matmul 18.5366 against
18.543 ms, BinaryNg 6.9675 against 6.972, LayerNorm 3.2957 against 3.296, SDPA 3.2599 against
3.315, ReshapeView 2.6317 against 2.629, NlpCreateHeads 2.4366 against 2.446, Permute 1.6111
against 1.600, Transpose 0.4958 against 0.498). That agreement is the check on the alignment.

### The three layout sites it exposes

| site | programs/step | ms/step | % of the 41.60 ms step |
|---|---|---|---|
| atom key-window build | 30 | **3.817** | **9.2** |
| token attention epilogue | 96 | 1.720 | 4.1 |
| AdaLN `s_terms` layer norm of `s` | 48 | 1.541 | 3.7 |

Per atom layer (6 a step), with the measured kernel duration of each program:

    reshape (1,140,32,128)->(1,280,16,128)    65.65 us   the W/2 half-window split, sub-tile
    permute             ->(1,16,128,280)      69.13 us
    matmul x keys_indexing                    98.42 us
    permute             ->(1,1120,16,128)    192.93 us
    reshape             ->(1,140,128,128)    206.66 us
                                             ------
                                             632.79 us

Per token layer (24 a step):

    o[:, :, :, :48]        Slice              11.37 us
    permute (0,1,3,2)      Transpose          11.18 us
    reshape ->(1,1,768,512) ReshapeView       41.66 us
    permute (0,2,1)        Transpose           9.07 us
                                              -----
                                              73.28 us

## 2. The atom key window is a contiguous slice, not a matmul

`boltz2.get_indexing_matrix` builds `index[k,j] = clamp(j - 2k + h/2, 0, h+1)` and keeps classes
1..h, so `M[j, k*h + c] = 1` exactly when `j = 2k - h/2 + 1 + c`. Substituting into
`single_to_keys`' einsum and folding the half-window index back into the atom index:

    s_kv[k, r, d] = flat[k*W + (W/2 - H/2) + r, d],   r = 0 .. H-1,  zero outside the sequence

The key window of query window k is the **H contiguous atoms starting H/2 - W/2 before it**. With
W=32 and H=128 that is `flat[32k-48 : 32k+80)`. So it is `H/W = 4` tile-aligned slices of one
shifted copy, concatenated -- no matmul, no half-window split, no permute.

`_atom_key_window` builds it that way, gated on W, H and the indexing matrix's shape and on
nothing else. Off-fold at the production shape [1, 140, 32, 128] bf16:

| | us | bit-exact vs the exact gather |
|---|---|---|
| shipped chain, bfloat4_b indexing | 934.53 | no, max abs 0.03125 |
| shipped chain, bfloat16 indexing | 939.97 | no, max abs 0.03125 |
| **window, 4 slices + concat** | **272.43** | **yes, torch.equal** |

The two indexing dtypes agree with each other **bit for bit** and both miss the exact gather by
the same 0.03125, so the loss is the **matmul's math fidelity truncating the value operand**, not
the one-hot's dtype. The window is the more accurate of the two: it reproduces `single_to_keys`
as the reference defines it.

## 3. On the step, profiler off

The profiler costs 3.81x on this block's wall and all of it is in the gaps, so the step is timed
bare: 5 blocks of 10 replays of one grabbed `Diffusion.__call__`, median block.

| arm | ms/step | spread over 5 blocks |
|---|---|---|
| base | **41.6014** | 0.05 % |
| `TT_BIO_ATOM_KEY_WINDOW=1` | **38.7190** | 0.09 % |

**1.07444x on the step, -2.8824 ms**, i.e. -480 us per atom layer over 6 atom layers. The base arm
reproduces `b2z2-sampler-stall-split`'s 41.4820 ms bare wall to 0.29 %.

**PROJECTED on Blackhole**, on the WH step ratio and the cell's committed 26.400 ms step:
**24.571 ms/step**. No Blackhole chip is held by this row and this is not a measurement.

## 4. What was screened and refuted

The token epilogue's four programs do NOT collapse into `nlp_concat_heads`: it runs in 21.36 us
against the chain's 73.74 but returns the wrong values (max abs 6.45), because it concatenates the
PADDED 64-lane head dimension where the model's head_dim is 48. `permute(0,2,1,3) + reshape` is
bit-exact and 308.17 us, 4x slower than the chain it would replace. Both in
`epilogue_probe_wh_c2.json`.

## 5. Cheat check

200 sampling steps, 3 recycles, the full 35-row MSA depth, seed 0 and `cdk2x2_512` are untouched.
The window gathers the same atoms into the same tensor and removes no model work; the step count
in every measurement here is the shipped one.

## 6. On the fold

`fold_ab.py --arms base,window,base --reps 2`, cold fold discarded, `base` at both ends of every
rep. 200 sampling steps, 3 recycles, `cdk2x2_512`, commit `6f5e7525`.

| | base | window |
|---|---|---|
| rep0 first / last | 42.406 / 40.876 s | 40.572 s |
| rep1 first / last | 41.358 / 42.403 s | 40.732 s |
| **median** | **41.880 s** | **40.652 s** |

**FOLD-RATIO-WH 1.03022x** by the median rule the protocol uses. The base arm's own spread is
**3.74 %** across four runs -- larger than the 1.23 s the ratio claims -- and base-first beats
base-last in one rep and loses in the other, so that spread is noise rather than drift. The
defensible bracket is **1.0075x min-to-min to 1.0302x median**, with the step measurement putting
the honest middle at **1.01396x** (2.8824 ms/step x 200 = 0.5765 s). Every window fold is faster
than every base fold (max window 40.732 < min base 40.876), the right direction at 1/15 by chance,
but two runs against four cannot carry a landing claim. **The number this row stands behind is the
step's 1.07444x**; the fold leg needs more reps before it can carry one of its own.

**PARITY: the fold output is bit-identical in both arms and in all six runs**, CIF sha256 prefix
`da476491dbb2a847`. The 0.03125 the two gathers differ by at the op does not reach the structure
at this fixture. That is one fixture, not a gate: the lever stays off by default.
