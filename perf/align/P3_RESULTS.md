# p3-align-widen — the alignment penalty is a tt-metal version effect, not a reshape guard

Phase 3, OPTIMISE. Protenix-v2, trunk only, 298 aa (token axis 298 logical, 320 padded; c_z=256).
SCOPE: protenix, trunk, 298. Branch `wk/protenix-trunk--p3-align-widen`, pushed. **Nothing merged.**

qb2 card 0, board 007 chip 0. Two ttnn builds on the one chip: the **0.68.0 production wheel** and a
**tt-metal source build** (`~/tt-metal-xtts`, HEAD `4910fe1e20`, upstream content of 2026-08-05,
newer than `v0.74.0-dev20260620`; the clone carries no tags so I cannot name its release). Everything
here is a **ratio on qb2 and owes a qb1 re-measurement at 0.67.4** before it drives a merge
(charter §4.8). Board-mate chip 1 carried no python process during any timed run (`ps aux` checked
before each), and `uptime` is recorded beside every measurement below.

**The headline, and it retires the lever this leg was authored to build:** the 1.585x the trunk pays
for a logically-298 contracted axis is **gone in newer tt-metal**. Same chip, same shapes, same
program config, both arms:

| | logical 320 | logical 298 | ratio |
|---|---:|---:|---:|
| ttnn **0.68.0** production wheel | 69.84 us | 109.81 us | **1.5724x** |
| tt-metal **source build** | 69.26 us | 69.48 us | **1.0031x** |

The aligned arm is the same on both (0.8 % apart against the 0.68 figure), so the newer build did not
get generally faster: it stopped charging for the unaligned axis specifically. All four output
tensors are **bit-identical across versions** — `torch.equal` True, 0 of 2 841 728 elements differ.

**And the second headline kills the obvious conclusion.** I extended the cross-version arm to the
other three sites in P2's blast radius, and **SDPA @1629 did not get the fix — it regressed.** At
each build's own best program config the newer build's SDPA is **1.76x slower** (1056.83 us against
1864.72 us at `q_chunk = k_chunk = 128`), and at the fold's own config it is 1.51x. SDPA is the
largest row in the trunk, so on these numbers **a version upgrade is a net loss of ~536 ms/fold**:
+349.5 gained on the matmul-class sites, −885.3 lost on SDPA. The right recommendation is therefore
the opposite of what the contraction row alone implies, which is why the other three sites were
worth the extra pass.

## Predictions (before measuring)

Committed in `5f43fdf9` and pushed before the device was opened (`perf/align/P3_PREDICTIONS.md`).

**P1 — `fill_implicit_tile_padding(x, 0.0)` costs 2-10 us/call** on `[1,32,298,298]` bf16 L1,
against P2's 22.51 us/call `multiply_` floor, because it writes only the padding lanes (~2.2 % of
6.554 MB). Wrong above 15 us/call. → **WRONG, badly.** 54.29 us/call, and **flat in size**.

**P2 — the fill has to land after the permute**, because at `tenstorrent.py:1347` the chunk is
`[1,L,L,C]` and only dim2 carries tile padding; the contracted axis is created by the permute. Wrong
if a mask-on arm is bit-exact against the aligned arm at the fold's own shape. → **CONFIRMED by the
live trace**, and the mask route is separately killed on cost.

**P3 — one fill does not serve 16 contraction calls per block.** Wrong if the padding of a permuted
chunk is already zero. → **WRONG.** `ttnn.permute` leaves 0.0 in the padding lanes it creates, and
`fill_implicit_tile_padding` is flat in size, so one 50.15 us fill on `x_norm_in` amortises over all
8 contractions in a trimul: **6.27 us per contraction**, the cheapest fill that exists.

**P4 — the net lands in 250-320 ms/fold** for the contraction row. → **CONFIRMED on the arithmetic**
(289.5 ms/fold at the amortised fill price) and **irrelevant in the outcome**, because the version
finding removes the penalty without any fill at all.

**P5 — the widen is a metadata change once the guard is relaxed**: same buffer address, no copy,
`torch.equal` True on the valid region. → **CONFIRMED exactly.**

**P6 — relaxing `reshape_common.cpp:50` alone is not enough**; the view-eligibility path will also
have to accept a logical shape that grows into padding. → **WRONG.** One line is enough, for both
`ttnn.experimental.view` and `ttnn.reshape`.

**P7 — the alignment does not survive the block.** → **CONFIRMED, and sharper than P2's source
trace**: it dies at the contraction's own output, inside the same chunk iteration.

**P8 — no fold-level number through the widen exists at the end of this pass**, because the source
build's python bindings are 3.12 and the tt-bio env is 3.10. → **CONFIRMED.** Stated plainly below
rather than projected.

## Roofs, measured on this card

Every roof below I measured myself on qb2 card 0 this pass (`perf/align/a_probe.py roofs`,
`perf/align/p3_roofs_qb2c0.json`, host load average 0.2-0.5). **I did not inherit any of them**
(charter §4.1). Grid reported by the device at open: **11x10 = 110 cores**.

| roof | measured here this pass | how |
|---|---:|---|
| square compute, HiFi4 bf16, `fp32_dest_acc_en` + `packer_l1_acc`, DRAM out | **105.45 TFLOP/s** | 4096³. Nothing is scored against it (charter §4.6) |
| **K=320, output width nt=10, L1 in and L1 out** — the contraction's own class | **52.30 TFLOP/s** | 10240x320 @ 320x320, best of 12 points; grid 11x10, `in0_block_w=2` |
| K=320, nt=10, DRAM out | 35.26 TFLOP/s | same shape, DRAM output |
| DRAM read | 323.39 GB/s | DRAM → L1 clone, 8 MB |
| DRAM write | 252.91 GB/s | L1 → DRAM clone, 32 MB |
| DRAM combined read+write | 367.22 GB/s | DRAM → DRAM clone, 32 MB |
| machine balance, **K=320-corrected, L1 out** | **142.4 FLOP/byte** | 52.30 / 367.22 |

The contraction moves 19.66 MB of L1 (3 x 6.554) for 2.097 GFLOP, so its **arithmetic intensity is
106.7 FLOP/byte**, below the K-corrected machine balance of 142.4 and far below the square 338. The
traffic is L1-resident, not DRAM, so neither DRAM roof binds it: **the binding roof is compute at
52.30 TFLOP/s.** Against it, on this card this pass:

| arm | TFLOP/s | % of the 52.30 TFLOP/s K=320/nt=10 L1-output compute roof |
|---|---:|---:|
| 0.68.0, logical 320 (aligned) | 30.03 | 57.4 % of that roof |
| 0.68.0, logical 298 (production) | 19.10 | **36.5 % of that roof** |
| source build, logical 298 | 30.18 | **57.7 % of that roof** |

**Cores engaged: 100 of 110.** `per_core_M = per_core_N = 1` at Mt = Nt = 10, so the 11th column
receives no work. P2 measured this by grid ladder (11x10 against 10x10, 1.3 % apart against the aligned arm's own figure, on the aligned
arm); I did not re-take that ladder this pass and I am citing it rather than claiming it. My own roof
search reproduces the same shape constraint: at grid 8x8 the circular buffers exceed L1 and only
`in0_block_w=1` allocates at all.

**Overlap: nearer `compute + comm` than `max(compute, comm)`, and I measured it myself this pass.**
`fill_implicit_tile_padding` costs **54.29 us/call standalone** and **53.18 us/call inserted into the
real `TriangleMultiplication` loop** — 2.1 % apart against the standalone figure. An op dropped into
this loop is charged in full; nothing hides behind a neighbour. P2's in-block-vs-standalone
contraction figures (113.94 against 113.51) say the same thing about the contraction itself.

## What changed, and the A/B that measured it

Three changes, all on the branch, none merged.

### C1 — the tt-metal guard, relaxed (release-gated, stays on the branch)

`ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_common.cpp`, one clause in
`infer_dims_for_reshape` (`perf/align/ttmetal_reshape_widen.patch`):

```cpp
-        TT_FATAL(new_volume == old_volume, "Invalid arguments to reshape");
+        // Equal logical volume is the wrong invariant for a tiled tensor. A logical shape that
+        // grows into padding the tensor already owns has exactly the same physical footprint --
+        // same pages, same tiles, same buffer -- so it is a metadata update, not a data movement.
+        bool widen_into_own_padding = ... // TILE layout, same rank, leading dims equal the padded
+                                          // shape, and round_up(new[-1]) == padded[-1], same [-2]
+        TT_FATAL(new_volume == old_volume || widen_into_own_padding, ...);
```

Built incrementally in `~/tt-metal-xtts` (EXIT=0), then measured:

| | result |
|---|---|
| `ttnn.experimental.view(x, (1,32,320,320))` on `[1,32,298,298]` L1 | logical **[1,32,320,320]**, padded [1,32,320,320], **same buffer address 1511424 in and out**, **1.85 us**, no device work |
| `ttnn.reshape(x, (1,32,320,320))` | identical, **1.59 us**, same buffer address |
| the widened arm's valid region against the production arm | `torch.equal` **True**, 0 of 2 841 728 differ |

**It is a metadata change, exactly as argued: same buffer, no copy, no page moved.** P6 was wrong —
one line does it, and `ttnn.reshape` reaches it too, so no second guard is in the way.

**And on the build where it is reachable it is worth 2.57 us/call, not 40.79**, because that build no
longer charges for the unaligned axis. 2.57 us/call at x524 x16 is ~21.5 ms/fold, and it is a probe
delta on a non-production ttnn, so it is not a delivered number.

### C2 — the fill, priced properly (deliverable 1)

`protenix.py:2223` calls `self.PF(s, z3)` with no mask, so the tail-zeroing
`ttnn.multiply_(a_chunk, mask_u)` at `tenstorrent.py:1347` never runs. Turning it on, in the real
module at the fold's own shape, on a quiet host:

| route, measured in the real `TriangleMultiplication` | us per trimul call | us per contraction | verdict against 40.79 saved |
|---|---:|---:|---|
| production mask passed at `protenix.py:2223` (`[1,L,L,1]` broadcast at 1347) | **+1109.6** | **138.70** | **KILLED**, 3.4x the saving |
| `fill_implicit_tile_padding` before each contraction | **+425.5** | **53.18** | **KILLED**, 1.3x the saving |
| `fill_implicit_tile_padding` on `x_norm_in`, once per trimul, amortised over 8 | **+50.2** | **6.27** | the only route that clears the bar |

Standalone ladder on the same card (`perf/align/p3_fillpad_qb2c0.json`), which is where the
amortisation comes from:

| shape | bytes | `fill_implicit_tile_padding` | `multiply_` 2-D mask | L1 clone |
|---|---:|---:|---:|---:|
| `[1,8,298,298]` | 1.64 MB | 54.24 us | 14.11 us | 9.23 us |
| `[1,32,298,298]` (the contraction operand) | 6.55 MB | 54.29 us | **22.46 us** | 12.79 us |
| `[1,64,298,298]` | 13.11 MB | 55.42 us | 40.78 us | — |
| `[1,298,298,256]` (`x_norm_in`) | 48.8 MB | **50.15 us** | 339.99 us | 247.05 us |

**P2's 22.51 us/call floor reproduces exactly (22.46).** The new fact is that
`fill_implicit_tile_padding` is **flat in size** — 50-55 us from 1.6 MB to 48.8 MB — so it is bound
by per-tile-row dispatch, not by the padding bytes. That is what makes the amortised placement work:
one 50.15 us fill on the 48.8 MB `x_norm_in` before the chunk loop, serving all **8 contractions per
trimul** and therefore **16 contraction calls per block**, is 6.27 us per contraction against 40.79
saved — net 34.5 us/call, **289.5 ms/fold**, inside my stated 250-320 band. P2's 153.3 ms/fold net
was priced at a per-call fill; the amortised fill nearly doubles it.

**Two caveats I am not glossing.** First, the zero has to survive `minimal_matmul` (contracts over
c_z, so a zero row stays zero), the 4-way chunk, and `multiply_` with a sigmoid activation (0 x
sigmoid = 0) — that chain is an argument on 0.68, not a measurement, because reading a padding lane
requires the widen. Second, on the source build I *did* read it: `ttnn.permute` leaves **0.0** in the
padding lanes it creates, and so does `ttnn.matmul`. Both point the same way, and neither is the
production chain with `layer_norm`'s beta in it.

### C3 — deliverable 3: where the alignment dies

Live shape trace of one real trimul call, every ttnn op wrapped
(`perf/align/p3_trace_qb2c0.json`, 86 records, 8 chunk iterations):

```
layer_norm  [1,298,298,256]            -> [1,298,298,256]   padded [1,298,320,256]
chunk       [1,298,298,128]            -> 4 x [1,298,298,32] padded [1,298,320,32]
multiply_   [1,298,298,32] x2          -> [1,298,298,32]
permute     [1,298,298,32]             -> [1,32,298,298]     padded [1,32,320,320]
matmul      [1,32,298,298] x2          -> [1,32,298,298]     padded [1,32,320,320]
permute     [1,32,298,298]             -> [1,298,298,32]     padded [1,298,320,32]
concat                                 -> [1,298,298,256]
linear      [1,298,298,256] @ [256,256]-> [1,298,298,256]
```

**A widen applied to a contraction operand cannot survive its own matmul.** The output comes straight
back to `[1,298,298,32]` and concats into a 298-logical pair tensor, so a fill or a widen has to be
re-applied per contraction — unless it is placed on `x_norm_in`, which is why that placement is the
only one that amortises. Before the permute, dim1 carries **no tile padding at all** (298 real rows);
the permute is what creates padding on both inner axes. That is P2's confirmed prediction and it
kills the mask site at 1347 on correctness grounds as well as on cost: it zeroes dim2's padding, and
for the non-ending trimul the contracted axis comes from dim2 while for the ending one it comes from
dim1, which has none.

**Placements P2 already killed and I did not re-test:** `ttnn.pad` on an operand (55.17 us/call),
`ttnn.pad` on `z` (332.16 us/call), and building `z` at 320 real rows (+4.0 % against each pair-shaped op's own baseline). All three remain killed; the brief says do not re-test them and I did not.

### C4 — the whole blast radius, cross-version, and the SDPA regression

P2's 398.9 ms/fold is four sites. Last pass I only re-measured the contraction, which left 56.9
ms/fold untested and — as it turns out — the recommendation wrong. All four, same chip, quiet host
(load average 0.01), each pair at a fixed padded shape with only the logical length of the
reduced axis moving:

| site | 0.68.0: 298 / 320 / delta | source build: 298 / 320 / delta |
|---|---|---|
| trimul contraction @1355 | 109.81 / 69.84 / **+39.97 us** | 69.48 / 69.26 / **+0.22 us** |
| `attn@v` @378 | 95.28 / 71.17 / **+24.10 us** | 72.55 / 73.16 / **−0.61 us** |
| `softmax` over an unaligned axis | 27.82 / 24.47 / **+3.35 us** | 30.36 / 28.89 / **+1.47 us** |
| **SDPA @1629**, the fold's own `q_chunk = k_chunk = 64` | 1636.41 / 1605.80 / **+30.60 us** | 2481.12 / 2429.80 / **+51.32 us** |

The 0.68 column reproduces P2 closely — `attn@v` +24.10 against P2's 23.74, SDPA +30.60 against
40.38, softmax +3.35 against 4.26 — so this is the same measurement, not a different one.

**The two matmul-class sites are fixed on the newer build and SDPA is not.** SDPA's ratio barely
moves (1.019x against 1.021x) while its **absolute** time rises by half. Before calling that a
regression I swept its own program config, because a version can simply want a different chunk:

| `q_chunk = k_chunk` | 0.68.0, logical 320 | source build, logical 320 | source / 0.68 |
|---:|---:|---:|---:|
| 32 | 3688.26 us | 5186.57 us | 1.41x |
| 64 (the fold's own) | 1605.19 us | 2430.97 us | 1.51x |
| **128** | **1056.83 us** | **1864.72 us** | **1.76x** |
| 256 | 1066.41 us | 1953.77 us | 1.83x |

**It is not a tuning artefact: the newer build is slower at every chunk size, and 1.76x slower at
each build's own best.** The regression is measured on bias-free SDPA arms, where the fold carries
an attention bias, and on one source snapshot rather than a release — both stated as limitations,
neither large enough to close a 1.76x.

**A finding that belongs to `protenix-trunk--p3-sdpa`, not to me.** On the 0.68 production wheel,
purely from the chunk size and at the production logical-298 shape, SDPA goes **1637.84 us at chunk
64 to 1159.58 us at chunk 128** — 1.41x against the fold's own configuration, on the wheel the trunk
actually runs. I measured it as a control for the version question and I am **not claiming it**; it
is theirs to verify against a real fold and to reconcile with chunk 320.

## Delivered ms/fold

**Zero, on the production wheel, and that is a measurement rather than a shortfall.** Every route
that is expressible in ttnn 0.68.0 is net negative, measured as a production-path A/B against the
unmodified module on the same card in the same session:

| production-path A/B, real `TriangleMultiplication`, 298 aa, 0.68.0 | trimul wall | delta |
|---|---:|---:|
| baseline, unmodified (control arm) | **8.335 ms** | — |
| with `fill_implicit_tile_padding` before each contraction | 8.761 ms | +5.1 % against the baseline |
| baseline, unmodified (second control, mask run) | 8.310 ms | — |
| with the production mask turned on at `protenix.py:2223` | 9.420 ms | +13.4 % against the baseline |

The probe figure and the production figure agree: the standalone fill is 54.29 us/call and the same
fill inside the module is 53.18 us/call, 2.1 % apart against the standalone figure. **There is no gap to report between the
microbenchmark and the block wall** — which is itself the finding, and it is the opposite of the
failure this org has had twice before.

For scale in the same units: the contraction is 8 calls x 109.48 us = 0.876 ms of the 8.335 ms
trimul wall (**10.5 % of the trimul wall**), and the alignment penalty inside it is 8 x 39.99 us = 0.320 ms
(**3.8 % of the trimul wall**). Two trimuls per block, 480 pf_stack block executions plus the MSA
stack's 40 and the confidence head's 4 = **x524** (charter §4.9), 16 contraction calls per block =
8384 calls/fold.

What is **available and not delivered**, each labelled with what it rests on. All of it is a per-call
delta scaled by the charter's conversion, not a fold A/B:

| lever | per call | calls/fold | ms/fold | status |
|---|---:|---:|---:|---|
| version effect, trimul contraction @1355 | +39.97 us | 8384 | **+335.1** | measured on a **source build**, not the production wheel |
| version effect, `attn@v` @378 | +24.10 us | 524 | **+12.6** | same |
| version effect, `softmax` | +3.35 us | 524 | **+1.8** | same |
| **version cost, SDPA @1629** | **−844.71 us** | 1048 | **−885.3** | same, and it is the biggest term |
| **net of a version upgrade on the trunk at 298 aa** | | | **−535.8** | **a loss**, on these arms |
| the guard relaxation, on the source build | +2.57 us | 8384 | 21.5 | measured, but the penalty it removes is already gone there |
| the guard relaxation backported to 0.68, with the amortised fill | +34.5 us | 8384 | 289.5 | **arithmetic, not a measurement** — needs a 0.68 source build |
| P2's blast-radius total, for reference | — | — | 398.9 | **a per-call sum**, never a fold A/B |

The SDPA row uses the fold's own program config on both builds (chunk 64, production logical 298:
2481.12 against 1636.41 us). At each build's own best chunk the gap is wider still, so this is the
conservative version of the number.

## Parity

Measured at the fold's own shape (`[1,32,298,298]` operands, padded `[1,32,320,320]`), never argued:

| comparison | verdict |
|---|---|
| widened + filled 320-logical arm vs the 298-logical production arm, source build | `torch.equal` **True**, max abs 0.000e+00, **0 of 2 841 728** differ |
| the contraction's output, **0.68.0 wheel vs source build**, logical 298 | `torch.equal` **True**, 0 of 2 841 728 differ |
| the same, logical 320 | `torch.equal` **True**, 0 of 2 841 728 differ |
| real trimul output, production mask on vs off | `torch.equal` **True**, 0 of 22 733 824 differ |
| real trimul output, `fill_implicit_tile_padding` per contraction vs unmodified | `torch.equal` **True**, 0 of 22 733 824 differ |

The cross-version row is the load-bearing one for the recommendation below: for this op, at this
shape, with this program config, the newer tt-metal is **arithmetically inert**. That is one op, not
a model-wide parity claim, and the release gate is what settles the rest.

## Merge recommendation

**Do not merge anything from this branch, and do not upgrade ttnn on this evidence.** Four
recommendations:

1. **Do not take a version upgrade for the alignment penalty.** It buys +349.5 ms/fold across the
   three matmul-class sites and costs **−885.3 ms/fold on SDPA**, so on the trunk at 298 aa it is a
   **net loss of ~535.8 ms/fold**. That is a per-call sum on one source snapshot, not a fold A/B,
   and it points the wrong way strongly enough that the burden is now on anyone proposing the bump.
   It also corroborates the existing verdict on the last upgrade this codebase evaluated.
2. **The tt-metal guard relaxation: hold, and offer it upstream.** It is correct, one line, a
   metadata change on the same buffer, and bit-exact — but on any tt-metal new enough to contain it
   there is no longer a penalty for it to remove, so its value to this org is ~21.5 ms/fold rather
   than 342. tt-metal is Tenstorrent's own repo and `new_volume == old_volume` is genuinely the
   wrong invariant for a padded layout, so the patch is worth filing on its merits. It is
   **release-gated and stays on `wk/protenix-trunk--p3-align-widen`**.
3. **The fill: do not ship it.** Every placement expressible on the production wheel is net
   negative, measured in the real module. The amortised `x_norm_in` placement (6.27 us/contraction)
   only pays once a widen exists.
4. **The one route worth costing is a 0.68 source build with the guard backported**, which is the
   only way to reach the matmul-class win **without** taking the SDPA regression: 289.5 ms/fold on
   arithmetic, needing a cold build at the production tag (hours) and then the amortised fill. Not
   attempted this pass.

**The production route, stated plainly.** qb1 runs the **0.67.4** wheel and campaign absolutes come
from there; qb2 and pc run 0.68.0 and produce **ratios only**. A source build at 0.68.0 does not
reach qb1's 0.67.4 wheel, and the source build I used is newer than either. Every number in this doc
owes a qb1 re-measurement before it drives a decision.

**Two things are open and both are cheap.** First, **which version carried the matmul fix, and
whether the SDPA regression arrived at the same one** — if they landed apart, an intermediate
version has the +349.5 without the −885.3, and that would change recommendation 1. Second, whether
the SDPA regression survives with the fold's attention bias present. The bracket did not come out
this pass for a concrete reason: `~/tt-metal` (`v0.73.0-dev20260610`) and `~/tt-metal-fused`
(`v0.74.0-dev20260620`) are both built but neither has a python env, and running them under the
xtts interpreter redirects the **python** package while the native dispatch-kernel source root
still resolves to the interpreter's own tree, so the JIT build mixes headers from one tree with
kernel sources from the other and the device refuses to open. **One venv per tree closes it**, and
that is the next pass's first job.

## Corrections to the inherited record

1. **The 1.585x is a tt-metal version effect, not a property of the hardware or of ttnn's design.**
   On the same chip, same shapes, same program config: 1.5755x on the 0.68.0 wheel, **1.0042x on a
   newer source build**. The aligned arm is unchanged across versions (69.49 against 69.86), so this
   is a targeted fix to the unaligned path, not a general speed-up. P2's mechanism (the NCRISC
   `reader_bmm_tile_layout_in0_sender_padding`) is consistent with it and stands.
2. **The guard was never the binding constraint it looked like.** Relaxing
   `reshape_common.cpp:50` works, is one line, and gives a genuine zero-copy widen — and buys 2.57
   us/call on the only build where it can be applied. The lever P2 identified is real and its price
   tag moved by a factor of 16.
3. **P6 was wrong: one guard, not two.** Both `ttnn.experimental.view` and `ttnn.reshape` widen
   correctly once line 50 accepts the case; there is no second view-eligibility check in the way.
4. **`fill_implicit_tile_padding` exists in the 0.68 wheel, and it is not the cheap op it sounds
   like.** 54.29 us/call on a 6.55 MB operand, against 12.79 us to clone the whole tensor — 4.2x the
   cost of rewriting every byte, to write only the padding lanes. It is **flat in size** (50-55 us
   from 1.6 MB to 48.8 MB), so it is dispatch-bound, and that flatness is the only reason an
   amortised placement beats P2's 22.51.
5. **The production mask route costs 138.70 us per contraction, not 22.51.** P2's 22.51 was a
   standalone `multiply_` against a 2-D mask (which I reproduce at 22.46). The mask production would
   actually pass is `[1,L,L,1]`, a subtile broadcast, and in the real module it costs **13.4 % of
   the trimul wall**. Deliverable 1's answer is that turning the existing mask on is decisively
   negative.
6. **P2's "one fill cannot serve 16 contraction calls" is wrong, and the fix is the fill's own
   flatness.** One `fill_implicit_tile_padding` on `x_norm_in` per trimul is 6.27 us per
   contraction. That moves the hypothetical net from 153.3 to **289.5 ms/fold** — for anyone who
   ever gets a widen on 0.68.
7. **`ttnn.permute` and `ttnn.matmul` leave 0.0 in the padding lanes they create**, measured through
   the widen on the source build. P2 flagged this as unknown. It does not change today's answer, but
   it is the fact any future fill placement should be designed against.
8. **398.9 ms/fold remains a per-call sum and has never been a fold A/B.** All four of its sites are
   now re-measured on both builds: the contraction at 335.1 ms/fold (39.97 us/call against P2's
   40.79, 2.0 % apart against P2's figure), `attn@v` at 12.6, `softmax` at 1.8 and SDPA at 30.60
   us/call. Nothing in it is untested any more.

9. **SDPA does not get the version fix, and it regresses.** Its unaligned ratio is unchanged
   (1.019x on 0.68.0 against 1.021x on the source build) while its absolute time rises 1.51x at the
   fold's own chunk and **1.76x at each build's own best chunk**. That single row is larger than
   the whole matmul-class gain and it reverses the recommendation the contraction row alone implies.
10. **`softmax` no longer pays for an unaligned axis on the newer build** (+3.35 us on 0.68.0
   against +1.47 us), and `attn@v` @378 goes from +24.10 us to **−0.61 us**. So the version fix
   covers the matmul class and the reduction, and misses the flash-attention kernel.

**Interaction with `protenix-trunk--p3-sdpa`, so the CTO does not double-count.** I **did**
re-measure SDPA this pass, on both builds, and the answer is that its alignment penalty is
version-invariant: +30.60 us/call on 0.68.0 and +51.32 on the source build, ratio ~1.02 either way.
So the version lever does **not** take SDPA's row — chunk size does. **Do not add my SDPA figure to
p3-sdpa's**: on the 0.68 production wheel, at the production logical-298 shape, going from the
fold's chunk 64 to chunk 128 takes SDPA from 1637.84 to 1159.58 us/call, 1.41x, which dwarfs the
30.60 us alignment delta and is p3-sdpa's territory. That row is theirs; mine is the contraction.

**Generalisation, recorded and not chased** (charter §1): a version that stops charging for an
unaligned contracted axis helps every model in this codebase that contracts over a non-tile-multiple
axis. One line, out of scope for this org.

## Artefacts

All on `wk/protenix-trunk--p3-align-widen`, pushed:
`perf/align/p3_probe.py`, `p3_widen.py`, `p3_xver.py`, `ttmetal_reshape_widen.patch`,
`reshape_common.patched.cpp`, and the JSON results `p3_roofs_qb2c0.json`, `p3_fillpad_qb2c0.json`,
`p3_trimul_qb2c0.json`, `p3_fillinplace_qb2c0.json`, `p3_trace_qb2c0.json`, `p3_widen_qb2c0.json`,
`p3_xver_0680.json`, `p3_xver_srcbuild.json`.

The patched `~/tt-metal-xtts` tree on qb2 still carries the relaxation and the rebuilt libraries;
the originals are beside them as `build_Release/lib/*.p3bak`.
