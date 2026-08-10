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
| ttnn **0.68.0** production wheel | 69.49 us | 109.48 us | **1.5755x** |
| tt-metal **source build** | 69.86 us | 70.15 us | **1.0042x** |

The aligned arm is the same on both (0.5 % apart against the 0.68 figure), so the newer build did not
get generally faster: it stopped charging for the unaligned axis specifically. All four output
tensors are **bit-identical across versions** — `torch.equal` True, 0 of 2 841 728 elements differ.

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
| 0.68.0, logical 320 (aligned) | 30.18 | 57.7 % of that roof |
| 0.68.0, logical 298 (production) | 19.16 | **36.6 % of that roof** |
| source build, logical 298 | 29.89 | **57.2 % of that roof** |

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

What is **available and not delivered**, each labelled with what it rests on:

| lever | per call | ms/fold | status |
|---|---:|---:|---|
| the version effect on the contraction row | 39.33 us | **329.8** | measured on a **source build**, not the production wheel. Not delivered |
| the guard relaxation, on the source build | 2.57 us | 21.5 | measured, but the penalty it removes is already gone there |
| the guard relaxation backported to 0.68, with the amortised fill | 34.5 us | 289.5 | **arithmetic, not a measurement** — needs a 0.68 source build |
| P2's blast-radius total (SDPA @1629, `attn@v` @378, `softmax`) | — | 398.9 | **a per-call sum**, never a fold A/B. Only the 342.0 contraction part is re-measured here |

I did not re-measure SDPA, `attn@v` or `softmax` on the source build, so **56.9 ms/fold of P2's
398.9 is unretested** against the version finding.

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

**Do not merge anything from this branch.** Three separate recommendations:

1. **The tt-metal guard relaxation: hold, and offer it upstream.** It is correct, it is one line, it
   is a metadata change on the same buffer, and it is bit-exact — but on any tt-metal new enough to
   contain it there is no longer a penalty for it to remove, so its value to this org is ~21.5
   ms/fold rather than 342. tt-metal is Tenstorrent's own repo and `new_volume == old_volume` is
   genuinely the wrong invariant for a padded layout, so the patch is worth filing on its merits.
   It is **release-gated and stays on `wk/protenix-trunk--p3-align-widen`**.
2. **The fill: do not ship it.** Every placement expressible on the production wheel is net
   negative, measured in the real module. The amortised `x_norm_in` placement (6.27 us/contraction)
   only pays once a widen exists, and by then the version that provides the widen has already
   removed the penalty.
3. **The real question this leg has surfaced is a dependency bump, and it is above my pay grade.**
   329.8 ms/fold of the trunk is sitting in a tt-metal version difference. The org should cost a
   ttnn upgrade, on evidence, rather than build a fill.

**The production route, stated plainly.** qb1 runs the **0.67.4** wheel and campaign absolutes come
from there; qb2 and pc run 0.68.0 and produce **ratios only**. A source build at 0.68.0 does not
reach qb1's 0.67.4 wheel, and the source build I actually used is newer than either. So there are
exactly two real routes to the 329.8 ms/fold, and neither is a patch to a wheel:

- **upgrade the ttnn wheel** to a version containing the fix, which is a dependency major-bump: it
  is release-gated, it needs the full parity gate on qb1 at the production shapes, and this
  codebase already has one recorded upgrade that was rejected on numerics for a smaller gain. My
  cross-version bit-exactness result covers **one op**, not the model.
- **build tt-metal from source at 0.67.4/0.68.0 and backport the guard**, which buys the 289.5
  ms/fold arithmetic above but leaves production on a source build. Cost: a cold build at that tag
  is hours, and it was not attempted this pass.

**Which version first carried the fix is open**, and it is the cheapest next question: `~/tt-metal`
(`v0.73.0-dev20260610`) and `~/tt-metal-fused` (`v0.74.0-dev20260620`) are both already built on
qb2, but neither has a python env of its own and running them under the xtts interpreter loads the
xtts libraries instead, so the bracket did not come out this pass. One venv per tree closes it.

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
8. **398.9 ms/fold remains a per-call sum and has never been a fold A/B.** This pass re-measured the
   contraction part of it (335.3 ms/fold at 39.99 us/call, against P2's 342.0 at 40.79 — 2.0 % apart against P2's figure) and left the other 56.9 untested against the version finding.

**Interaction with `protenix-trunk--p3-sdpa`, so the CTO does not double-count.** My SDPA @1629 row
is P2's 42.3 ms/fold and I did **not** re-measure it this pass, on either build. If the version
finding is taken up, SDPA's alignment penalty may vanish with the contraction's; if chunk 320 lands
first, SDPA's k-chunking changes and the row moves for a different reason. **Do not add my 42.3 to
p3-sdpa's figure** — whichever lever lands first takes that row, and the other one then owes a
re-measurement rather than a second claim on the same milliseconds.

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
