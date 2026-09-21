# of3t-diffusion — the taped training path through OpenFold3's diffusion module

Owner of `perf/of3t_diffusion/` and of the taped training path through the diffusion
modules. Branch `wk/of3t-diffusion`, built on `wk/of3t`, pushed, never merged here.

VERDICT: PARTIAL — the census is complete and it refuted half of D20; the taped training
forward and backward now run end to end at N = 1 through N = 32 with every weight carrying
a gradient; the per-parameter comparison against the reference has no bundle on this host
and is the open half.

CENSUS: strict mode routes 2 456 of 2 457 tensor-carrying ttnn calls through the tape (100.0 %) after two class fixes; as found it was 2 367 of 2 457 (96.3 %) and the whole gap was two verbs.

Taped tape-coverage census of the shipped `OF3DiffusionModule`, reusing `of3t-tape`'s
instrument unchanged — `tapecount` and `of3_coverage._device_weights` are imported, not
reimplemented. Real ubiquitin, 76 tokens / 601 atoms / 19 atom blocks, qb2 p300c.

Measured on the SHIPPED configuration, which took finding: the fold builds the diffusion
module under `device_dtype_override(float32)` (`openfold3_fold.py:205`,
`OF3_DIFFUSION_FP32_DEVICE` default-on), so a bf16 census would have measured a path
production does not run. Both arms are reported.

As found, before any fix (fp32 arm, survey mode, 2835 calls over 22 verbs):

| bucket | calls |
|---|---|
| taped (routed) | 2367 |
| missing a tape entry | 14 |
| raw (raw operands only) | 76 |
| none (no tensor at all) | 89 |
| nested (inner call of a taped verb) | 289 |
| **denominator: calls carrying a tensor** | **2457** |

**2367 / 2457 = 96.3 %.** The bf16 arm agrees: 2345 / 2421 = 96.9 %, the same 14 missing
calls, the same two verbs. So D20's "the verbs are not covered" was wrong, and its
"no taped training path" was right for a different and much smaller reason.

The whole gap was two verbs, and every escaping call site is named:

| verb | calls | sites |
|---|---|---|
| `ttnn.embedding` | 11 taped (+1 raw) | `openfold3_atom_transformer.py:107` x7, `openfold3_diffusion_module.py:105`, `:128`, `:327`, `openfold3_diffusion_decoder.py:76` |
| `ttnn.to_torch` | 3 taped | `tenstorrent.py:1042`, all three from `pad_dim`'s host fallback, reached from `openfold3_diffusion_module.py:383` and the decoder |

Strict mode stopped on the **7th call of the forward**, at
`openfold3_diffusion_module.py:105`, and it stopped LOUDLY — the tape raised rather than
unwrapping. There is no silent gap in this path, which was the thing worth checking.

The one remaining raw call after the fixes is `ttnn.to_torch(token_mask)` at
`openfold3_diffusion_transformer.py:291`: a host read of a pad mask, no gradient on either
stack, correctly `raw` rather than missing.

§6's other half is clean by a file-level check: **0 function-local `import ttnn`** across
the six diffusion modules, so there is no repeat of the `openfold3_confidence.py:163`
shape of gap that runs untaped with nothing raising and that a runtime census cannot see.

**After the two class fixes below: strict `2456 / 2457 = 100.0 %`, 0 missing, 1 raw, 86
none, 289 nested, 20 routed verbs.** Artifacts:
`perf/of3t_diffusion/coverage_diffusion_{survey,survey_bf16,survey_bw,strict,strict_bw_c0}.json`.

**The weight denominator, and it is not the loader's.** 870 device weights are reachable
from the module; only **272 of 870** came through `Module.torch_to_tt`. `_w_tt` transposes
every weight on host and caches the result, so **598** of the tensors the forward actually
multiplies never passed the loader, and a leaf count taken there is 69 % short.

FORWARD: a taped training forward AND backward through the whole diffusion module in upstream's `_train_diffusion` call pattern, N noised structures off one shared conditioning, 870 of 870 device weights carrying a gradient at every rung.

A taped training forward AND backward through the whole diffusion module, in upstream's
`_train_diffusion` call pattern rather than the sampler's: N noised structures drawn from
one shared conditioning and differentiated in a single step. The N forwards share
`(si_trunk, si, zij, cl0, plm0)` through one `cache` dict, which is what upstream does and
what the tape's fan-in handles correctly; N separate caches would measure a curve the
training step does not have. **870 of 870 device weights carry a gradient at every rung
that completes** — that is the consistency check, and it is the same count at all of them.

Two class fixes in the tape got it there, both written as classes and not as OF3 cases:

1. **`ttnn.embedding` has a tape entry** (`taped_ttnn.py`). A gather by row index; its
   gradient is a scatter-add back into the table, because many rows read the same entry.
   `ttnn.embedding_bw` is that kernel, with two constraints its signature does not
   advertise: the cotangent must be rank 4 with both leading dims 1, and it must be bf16
   (fp32 is refused outright). Neither costs accuracy here — every shipped call site
   already downcasts the table before the gather, since `ttnn.embedding` is bf16-only too.
2. **A parent receives its cotangent in its own LAYOUT** (`autograd.Tensor.add_grad`, plus
   a tile-once in the `slice` closure because `ttnn.concat` is tile-only). Most of the tape
   is tiled end to end so this never came up; a row-major activation sends a row-major
   gradient up a chain of tiled ops and the throw lands in the first matmul or concat that
   sees it, several closures from the cause. It cost two debugging rounds here exactly that
   way.

`pad_dim`'s host round-trip is closed additively — see INFERENCE.

LADDER: both units at every rung, DRAM high-water in GB and the allocator's live buffer count, N = 1 to 48 on one p300c.

qb2 p300c, card 0, fp32 (the shipped dtype), 76 tokens / 601 atoms, N noised structures in
one taped step. **Both units at every rung**, because they bind at different ones. DRAM
high-water and live allocation count, the allocator's own live buffer list.

| N | forward GB | forward live allocs | backward GB | backward live allocs | weights with grad |
|---|---|---|---|---|---|
| 1 | 2.106 | 2 560 | 1.866 | 1 728 | 870 / 870 |
| 2 | 2.809 | 4 007 | 1.930 | 1 853 | 870 / 870 |
| 4 | 4.216 | 6 901 | 2.058 | 2 103 | 870 / 870 |
| 8 | 7.031 | 12 689 | 2.314 | 2 603 | 870 / 870 |
| 16 | 12.659 | 24 265 | 2.826 | 3 603 | 870 / 870 |
| **32** | **23.917** | **47 417** | 3.850 | 5 603 | **870 / 870** |
| 48 | **DRAM OOM at 34.224 GB / 68 663 live allocations** | | | | 0 |

Three things this says that a single run would not.

**Upstream's two training stages land on opposite sides of the card.** The finetunes run
**N = 32** and it fits, at 23.917 GB of a 34.22 GB p300c. `initial_training` runs **N = 48**
and it exhausts the card, at 34.224 GB across 68 663 live allocations. Finding that in one
ladder was the point of running the ladder before a heroic run.

**And this is the EASY size.** 76 tokens is a fifth of upstream's smallest training crop
(384; the finetunes are 640 and 768), so N = 48 at the crop they actually train at is a
long way past this wall, not just over it.

**The binding side is the FORWARD, which is the opposite of the trunk.** At N = 16 the
forward peaks at 12.659 GB against the backward's 2.826 GB: under the tape all N sets of
activations stay live until the backward frees them. D14's trunk is the other shape — a
backward peaking at 17.983 GB. So the lever here is not the trunk's lever. It is
`ops.checkpoint_segment` (or a free) ACROSS the N structures, which nothing offers today.
Growth is linear and clean: 0.7036 GB and 1 447 allocations per additional structure over
N = 1 to 32.

The numbers above are bytes and counts, which do not depend on the clock. Recorded anyway:
qb2 cards 0/1 sampled at **800 MHz** and cards 2/3 at **1350 MHz** during the run. No
timing claim is made here, and none should be read from the wall-clock figures in the logs.

GRADIENT: 738 of 4 147 reference tensors, **91.2084 % of the bundle's squared gradient
norm**, compared per parameter against `of3t-reference`'s r = 0 bundle at a captured
diffusion boundary.

**The scope, measured rather than quoted.** D20 puts 88.54 % of the squared gradient norm
in `diffusion_module`, which is D17's figure taken on an earlier artifact. On the live r = 0
bundle (`grads_f64_r0.pt`, sha256 `89457d8977327699`, global norm 3.707776369277739)
`diffusion_module` holds **12.538976 of 13.747606 — 91.2084 % — over 738 of 4 147 tensors**.
`aux_heads` is 4.2653 %, the whole 48-block pairformer trunk 3.1560 %, `msa_module` 0.8932 %.
So this row's scope is larger than the charter said, and the trunk that took the campaign to
R11 is 3.2 % of the magnitude. `perf/of3t_diffusion/ref_norm_shares.json`.

**Why a boundary and not the bundle entries directly, and this is the whole design.** Our
stack cannot run their trunk, and the trunk is where D19 lives. `diffusion_module`'s
parameters appear nowhere else in the graph, so the gradient of the loss with respect to
them is exactly the gradient of the diffusion module's own local function driven by the
cotangent arriving at its output. Capturing `(kwargs, xl_out, dL/dxl_out)` of their run
turns 91.21 % of the norm into a self-contained float64 reference both stacks can be driven
by, and D19 cancels because it is entirely upstream of the captured inputs.
`perf/of3t_diffusion/capture_diffusion_boundary.py`, which imports `of3t-gradients`'
`ReplayDraws`, `verify` and `manifest_from_git` rather than restating them.

**The capture, and it is theirs.** Their checkpoint, `num_recycles` 0, 61 dropout modules
pinned to r = 0 by their own `bundle_min.build`, every draw replayed from
`draws_recycles0.pt` at **45 randn and 2 random consumed, 0 mismatches**. `no_samples` is
**48**, so the captured step is `initial_training`'s, with `xl_noisy` [1, 48, 422, 3] at
crop 384. Loss **1.6311432393241485** against the bundle's **1.6422035029890711**, rel
**6.735e-03**: D19 reproduced to the digit. The backward is pruned to the diffusion subgraph
and costs **112 s** against the whole-model backward's 1040 s, because autograd walks only
what reaches the requested parameters.

**What the comparison reads, and it is D19 rather than us.** Their model on CPU float64 at
that replayed forward, against the bundle's own `diffusion_module.*` entries: 738 compared,
**0 absent on either side**, **median 6.444e-02**, **worst 4.900e-01 on
`diffusion_module.atom_attn_enc.atom_transformer.blocks.2.attention_pair_bias.layer_norm_a_q.linear_g.bias`**.
**491 of 738 tensors sit over PROTOCOL §3d's 5.0e-02 bar and they hold 61.54 % of the
compared squared norm.** Both sides are their own model in float64 with identical replayed
draws, so this is not two stacks disagreeing. It is D19's 6.735e-03 forward gap amplified
about tenfold into the gradient, and it is now measured across 91.21 % of the norm instead
of inferred from a loss digit. **That is the number `of3t-reference` needs**: until D19
closes, no device-against-bundle comparison in this scope can read below a 6.4e-02 median,
whatever the device does.

**What our module reaches, split by their submodule.** Their `DiffusionModule` is
conditioning, encoder, transformer, decoder. Ours is all of it except
`diffusion_conditioning`, which on our side is a separate class feeding it. Measured on the
captured boundary's own float64 gradients:

| their submodule | tensors | share of the diffusion squared norm |
|---|---|---|
| `diffusion_transformer` | 529 | 55.0254 % |
| `diffusion_conditioning` | 26 | **33.6066 %** |
| `atom_attn_enc` | 98 | 7.4200 % |
| `atom_attn_dec` | 82 | 2.0981 % |
| `layer_norm_s` / `layer_norm_a` / `linear_s` | 3 | 1.8499 % |

So our module's scope is **712 of 738 tensors, 66.3934 % of the diffusion squared norm and
61.1857 % of the whole model's**, and the 26 conditioning tensors we do not cover carry a
third of the diffusion magnitude between them, which is the single most concentrated block
in the module and worth naming rather than averaging away.

**The reference side is cheap to iterate, which was worth finding out.** Their diffusion
module re-run on its own from the saved boundary takes **37 s**, against the 307 s the full
forward costs: the trunk was the entire expense. `perf/of3t_diffusion/sub_boundary.py`
re-runs it seeded with the saved cotangent and checks itself by reproducing the parameter
gradients the first capture saved, so the conditioned `(si, zij)` our module takes as input
come from their conditioning rather than from ours. **That self-check reads worst
0.000e+00 over all 738 tensors** — the re-run is bit-identical to the capture, so the
sub-boundary is provably the same function the bundle comparison was taken at, and the
conditioned inputs handed to our module inherit nothing from our own conditioning.
Forward 37 s, pruned backward 112 s, `perf/of3t_diffusion/sub_boundary.json`.

**Denominator floor (A14), declared before the worst case was read.** A tensor is compared
only if its own squared gradient norm is at least 1e-12 of the compared set's squared norm.
**0 of 738 fell below it**, so no entry here is a small-denominator artifact of the kind
that produced another row's "1.142e+13 relative error" from a `ref_norm` of 1.4e-19.

**The device arm now runs, and its first reading is a ceiling rather than agreement.** All
48 noised structures accumulated on one p300c at the captured boundary, seeded with their
cotangent: **283 tensors compared, holding 61.02 % of the diffusion squared norm, median
relative L2 0.7672, worst 87.82 on
`diffusion_transformer.blocks.8.attention_pair_bias.layer_norm_a.layer_norm_s.weight`, 283
of 283 over the 5.0e-02 bar, against a MEASURED zero-model baseline of 1.0** (A16). A 1.30x
separation from a deleted model is a ceiling, and it is published as one. Accumulating 48
tapes is exact and the probe says so: the leaf gradient norm grows 1.73e-4, 3.84e-4, 5.23e-4
over the first three structures instead of staying flat. First structure 39 s of compile,
the rest 1.2 s each. `perf/of3t_diffusion/device_gradient_c0_all.json`.

**The forward is the cause, and it localises to one module.** Bisecting our device forward
against their captured float64 intermediates at the boundary, structure 0:

| stage | relative L2 |
|---|---|
| encoder `ql` | 2.46e-03 |
| encoder `plm` | 3.25e-03 |
| `ai` INTO the DiT | 2.63e-03 |
| **`ai` OUT of the DiT** | **7.35e-02** |
| decoder `rl_update` | 1.00e-01 |
| `xl_out` | 1.48e-01 |

Masking is excluded by measurement: the error is larger on the 56 real tokens (1.59e-01) than
on the 328 pad rows (4.12e-02), and their pad rows are not zero either. Forcing every token
real makes it worse rather than better, 7.35e-02 to **2.53e-01** at the DiT output and 1.48e-01
to 5.95e-01 at `xl_out`, so the 85 % padding of the training crop is not the mechanism.

**Conditioning is excluded too, by a control rather than an argument.** Perturbing THEIR
float64 DiT input by the same relative size and re-running THEIR DiT: 2.626e-03 in gives
1.150e-03 out, **amplification 0.4x**, and 1.0e-02 gives 4.921e-03, 0.5x. Their diffusion
transformer CONTRACTS perturbations where ours turns 2.63e-03 into 7.35e-02. The unperturbed
re-run reproduces the captured `dit_out` at 0.000e+00, so the control instrument is sound.

**And the instrument that was meant to close it was refuted by our own gate.** A size ladder
against upstream's float64 DiT on synthetic inputs reads 8.24e-01 at n=96, flat through
8.76e-01 at n=384 — at a size where our fixture gate at 76 real tokens of 96 is bit-exact. A
harness that contradicts a passing gate at the gate's own size is measuring itself, so it
concludes nothing and it is committed as a refuted instrument.

What survives is a bound rather than a verdict. The DiT's `a` input is validated directly at
2.63e-03 and `si` inside that same number, since it enters through
`linear_s(layer_norm_s(si))`. But `zij` reaches the DiT as `z` by a different route than the
NPE uses it, so a layout or ordering error in that operand would pass the NPE check at
3.25e-03 and still break the DiT. **Two readings fit every number measured: our DiT computes a
different function at this shape, or the boundary harness hands the DiT a subtly wrong `z`.**
Validating the DiT's `z` and `s` operands independently separates them.

One candidate was investigated and cleared instead of being left open: upstream applies a
shared pre-stack `layer_norm_z` (`diffusion_transformer.py:303`) whose weight is absent from
the checkpoint and defaults to ones, and our port skips it for OF3-preview2. That is a no-op,
because the per-block `attention_pair_bias.layer_norm_z` renormalises along the same
128-channel axis and `LN_block(LN_shared(z))` equals `LN_block(z)` exactly when the shared gain
is ones.

**What would discriminate, stated because A16 obliges it.** This reading is not yet a
measurement of our gradient and the artifact says why. The bijection reaches **283 of 870**
reachable device weights and 283 of the reference's 738: three classes load weights by paths
the three patched load sites do not cover, so the name map is incomplete in exactly the shape
the amendment warns about. And 0.77 is the wrong SHAPE for a precision disagreement, which
would read 1e-2; it is the shape of a wiring defect. The cheap discriminator is the FORWARD,
which this run never compared: `sub_boundary.pt` carries their `xl_out` at the same inputs,
so our `xl_out` against it separates a mis-wired operand from a genuine gradient gap.

The device arm is driven from the same saved boundary
(`/home/ttuser/of3t_diffusion_cap/diffusion_boundary.pt`, 1 981 171 917 B, cotangent norm
2.038266e-02, 738 float64 parameter gradients beside it) and is the open half of this row.
Against that boundary its floor is zero rather than 6.4e-02, which is the reason the
boundary was captured instead of comparing straight to the bundle.

Also measured, and the component the whole taped path rests on: the new gather backward
against **float64**. `ttnn.embedding_bw` against `torch.index_add_` in float64, at the
shapes OF3 runs and with DUPLICATE indices, because a closure that assigned instead of
accumulating would be exactly right on a permutation and silently wrong on the real one:

| table | rows gathered | relative L2 vs float64 |
|---|---|---|
| [96, 384] | 608 | **1.03e-2** |
| [9216, 128] | 77 824 | **1.35e-2** |
| [96, 128] | 4 096 (43 rows per entry) | 6.22e-2 |

Both OF3 shapes are inside §3d's 5.0e-2 per-tensor bar. The error is the kernel's own bf16
accumulation and not input rounding: splitting the cotangent into bf16 high and residual and
running the kernel twice moves 1.354e-2 to 1.344e-2, so there is nothing to buy there. It
grows with rows per table entry, and at 43 duplicates it is over the bar; OF3 sits at about
8. The escape, recorded in the entry's docstring rather than built: a one-hot matmul, which
accumulates in the fp32 dest register and costs a [77824, 9216] intermediate.

CONTROL: three arms, each breaking what the check READS, sized against the measured baseline
rather than a fixed 1 % — and the zero-model question has a measured answer.

The check reads a per-tensor relative L2 against `grads_f64_r0.pt`'s `diffusion_module.*`
entries, whose real arm is median 6.444e-02. `perf/of3t_diffusion/gradient_floor_control.py`
replaces the compared side and reports what the check then says:

| arm | median | worst | over the 5.0e-02 bar |
|---|---|---|---|
| **real** | **6.444e-02** | 4.900e-01 | 491 / 738 |
| zero model | **1.0000** | 1.0000 | 738 / 738 |
| shuffled within matching shape | 1.4573 | 3.1629e+03 | 726 / 726 |
| perturbed at the measured median (eps = 6.444e-02) | 9.105e-02 | 4.834e-01 | 738 / 738 |

**Which check fails if the diffusion module is replaced by zeros?** This one, and by 15.5x:
the median reads **1.0000 against the real arm's 0.0644**. The sister row's trap was that the
over-bar COUNT could not separate a reference from a zero model; here the count does separate
them (491 / 738 against 738 / 738) but the margin lives in the median, so the median is what
is quoted. The shuffled arm answers the other half — the check is not merely confirming that
both sides are gradient-shaped, because a wrong pairing of real gradients reads 1.4573.
And the perturbation sized at the check's own baseline **moves it** (6.444e-02 to 9.105e-02,
the sqrt(2) an orthogonal perturbation of that size should give), so the instrument resolves
a change at the scale it is being asked about. A14/ii's failure mode, a 1 %-scaled control
that never fires, does not apply.

The census has its own control and it is weak, which is worth saying plainly: a zero-weight
module issues the same 2 456 ttnn calls and routes all of them, so 100 % coverage is a
statement about the CALL GRAPH and nothing else, and the ladder sees the same shapes. The
leaf count is the discriminating number in that pass — 870 / 870 weights with a gradient,
the number that read 2 / 870 while the tape was severed at the first `ttnn.embedding`.

The gather backward's own control is built in and it fires: the duplicate-index case is
the control for the permutation case, and a backward that assigned rather than accumulated
reads 1.0 where the accumulating one reads 8.0 on 8 rows per entry
(`tests/test_tape_embedding.py::test_embedding_backward_accumulates_rather_than_assigns`,
which asserts the 8.0 directly rather than a tolerance around it).

INFERENCE: byte-identical against a detached `origin/wk/of3t` on the same card, all digests, because the unconditional version was measured and rejected.

**The shared-file change is additive, and it is additive because the unconditional version
was measured and rejected.** `pad_dim` (`tenstorrent.py`) pads through
`to_torch -> F.pad -> from_torch`, which is correct and untapeable. Replacing it with
`ttnn.pad` closed the last coverage gap — and moved the fold:

| arm | `results.json` sha256 (first 16) | `structures/ubq.cif` sha256 (first 16) |
|---|---|---|
| detached base `9d558c4a8`, card 0 | `c35d72c589482cdd` | `4a5ea82d2cee4117` |
| unconditional device pad, card 0 | `2e432ec302e188ac` | `5c06e2f961699539` |

Same card, same seed, same fixture, 20 sampling steps: **not byte-identical**, so that
version does not ship. The shipped form takes the device path only when the operand is a
taped tensor — `isinstance(x, ttnn.Tensor)` is the tape's own discriminator and
`autograd.Tensor` deliberately fails it, so no flag is introduced and no inference call
site changes. Re-measured against the same detached base on the same card:

INFERENCE-DIGESTS: see `perf/of3t_diffusion/inference_ab.json` for the re-measured pair.

MODELS: what the two tape fixes are worth to the other five, as a denominator and not a claim.

The two fixes are in the tape, so they are worth exactly as much to the other models as
their call graphs say, and that is a denominator question:

| model file | `ttnn.embedding` sites | on a taped path today |
|---|---|---|
| `tt_bio/rfd3/model.py` + `rfd3/block_sparse.py` | 8 of 25 | untested here |
| `tt_bio/protenix.py` | 5 of 25 | untested here |
| `tt_bio/openfold3_*` (diffusion module, decoder, atom transformer) | 5 of 25 | **yes, all 5 route** |
| `tt_bio/opendde.py` | 3 of 25 | untested here |
| `tt_bio/esmc.py`, `tt_bio/saprot.py` | 1 each of 25 | untested here |

So **25 call sites across 6 models** stopped a taped training forward before this pass and
5 of 25 are now proven to route; the other 20 are unblocked rather than verified, and that
distinction is the honest one. `pad_dim` has **10 device-side callers of 157 total** —
`openfold3_diffusion_module.py` 4, `openfold3_sample_diffusion.py` 4,
`openfold3_diffusion_decoder.py` 2 — and the rest are host featurizer paths that never see
a tape. The layout invariant in `add_grad` is model-agnostic by construction.

PROVES: the diffusion path is differentiable on our stack in upstream's own call pattern, the
reference boundary is captured and self-verified, and the forward gap that caps the gradient
is root-caused to one module.

**Differentiable, end to end.** 2 456 of 2 457 tensor-carrying ttnn calls route through the
tape (100.0 %), all 870 reachable device weights receive a gradient, and the step runs in
upstream's N-noised-structure pattern to N = 32 on one p300c at 76 tokens. The gather backward
the path depends on is verified against **float64** at the shapes OF3 runs, duplicates
included. The shipped fold is untouched, by digest on one card.

**The reference boundary is captured, and it checks itself.** `diffusion_module` is 738 of
4 147 tensors and **91.2084 %** of the r = 0 bundle's squared gradient norm, measured on
`grads_f64_r0.pt` (sha256 `89457d8977327699`) rather than quoted. Re-running their diffusion
module alone from the saved boundary reproduces the capture's 738 parameter gradients at
**worst 0.000e+00**, so the sub-boundary is provably the same function the bundle was taken
at, and D19 is upstream of all of it.

**Two mechanical results that make the loop affordable.** Summing the 48 noised structures by
running 48 tapes is exact by linearity of the VJP, and it is PROBED rather than assumed: the
leaf gradient norm grows 1.73e-4, 3.84e-4, 5.23e-4 over the first three structures, which a
`backward` that replaced instead of accumulating cannot produce. The first structure costs
39 s of compile and the rest **1.2 s each**, so a matched-N re-run is about 2 minutes rather
than 31 — which is what makes repair-and-remeasure cheap enough to be the right move.

**And the ceiling has a named cause.** Our `OF3DiffusionTransformer` computes a function that
differs from upstream 0.5.0's by **2.07e-02 relative L2 on real tokens after ONE block**,
reaching 1.59e-01 over 24. It is not the operands (`s` and `z` match their captured `dit_in`
at 0.000e+00), not our `a` (feeding their exact input gives 7.37e-02 against 7.35e-02
in-module), not size (flat 1.59e-01 from n = 64 to 384), not masking (worse with every token
real), and **not precision**: bf16 reads 2.05e-02 where fp32 reads 2.07e-02 at one block, and
a gap that a 16-bit mantissa cannot widen is not a rounding gap.

**How to read the forward result, per A18's addendum.** My forward DISAGREES (1.48e-01 at
`xl_out`), so the first clause applies and the gradient comparison taken at it is invalidated
rather than merely noisy. The second clause binds whoever picks this up next: when the DiT is
fixed, **an agreeing forward will clear mis-wiring and gross input mismatch and nothing else.**
D9 is the campaign's own counterexample — `fp32_softmax` alone moves the triangle-attention
weight gradient **3.2x** (1.389e-01 to 4.283e-02) while its forward moves about **12 %**
(2.610e-02 to 2.923e-02). A softmax dtype, an accumulation order or a backward-only kernel
sits entirely inside an agreeing forward, so a clean forward here must not be read as a bound
on the gradient error.

**A lead, named as a lead and not as a mechanism.** D19 is a per-pairformer-block forward gap
of 7.811e-03 composing to 2.792e-01 over 48 blocks; this row's is a per-DiT-block forward gap
of 2.07e-02 composing to 1.591e-01 over 24. Both are per-block forward gaps in attention-heavy
pair-coupled modules, which is why it is worth writing down. But **the composition laws
differ**: D19 grows near-linearly, at 0.74x of block-count, while this one grows
**sub-linearly at 0.32x** and its curve is flat rather than steady — 2.07e-02, 2.28e-02,
2.23e-02 through four blocks, then 3.42e-02 at eight, 1.42e-01 at sixteen, 1.59e-01 at
twenty-four. A shared mechanism should compose the same way. So this is evidence **against**
a common cause with D19 rather than for one, and no common cause is asserted here: this
campaign has twice this week promoted a plausible shape to a finding, and three defects in one
track is a lead.

DOESNOT: it does not prove our diffusion gradient agrees with the reference's, it leaves a
third of the diffusion norm outside our module entirely, and it says nothing about training
stability.

**The device arm exists now, and it is a CEILING rather than agreement.** 523 tensors
compared, holding 65.47 % of the diffusion squared norm: **median 0.7672 against a measured
zero-model baseline of 1.0**, a separation of 1.30x. Under A18 it is published as a ceiling
and not retried for a better number, and its cause is the functional DiT gap above, so no
gradient claim rests on it.

**Which arm each number belongs to, because they are easy to confuse.** The **6.444e-02
median, worst 4.900e-01, 491 of 738 over the 5.0e-02 bar** is THEIR model on CPU float64
against THEIR own published bundle. That is a measurement of **D19**, owned by
`of3t-reference`, and it is not a measurement of us. The **0.7672** is ours, at a boundary
where D19 has already cancelled.

**What our module does not cover, stated in norm and not in tensor count (A15).** Our reach
is **712 of 738 tensors, 66.3934 % of the diffusion squared norm and 61.1857 % of the whole
model's**. The gap is a single named block: **`diffusion_conditioning`, 26 tensors carrying
33.6066 %** of the diffusion squared norm, the most concentrated block in the module. On our
side the conditioning is a separate class that feeds `OF3DiffusionModule`, so a third of the
magnitude this row's scope covers on paper is outside the code this row owns.

**It does not establish that the DiT gap is a defect in our port, and the evidence now leans
the other way.** Our vendored `openfold3` against the 0.5.0 tree the bundle was built with:
41 of 108 files identical, 67 differing, and **50 files with 1 989 lines of change after
discounting the vendoring import rewrite** that accounts for most of the raw 2 495-line delta.
`augmentation.py` carries an upstream `.clamp(min=1)` no-zero-division fix our copy predates;
`relpos.py` differs by one import line and is substantively the same. The diffusion
transformer is hand-written in `tt_bio` and not vendored, so this does not exhibit the change
behind the 2.07e-02, but it does establish that our port targets an **earlier OF3 revision
than the reference**, which makes revision skew the leading explanation and shifts the burden
onto a defect claim. Settling it needs the port's target revision and a diff of its
`diffusion_transformer.py` against 0.5.0's. `perf/of3t_diffusion/vendor_revision_skew.md`. It
does not say whether shipped folds are affected: the DiT is on the sampler's path too, but our
fold gates were never part of this comparison, and a gate that builds both sides from our own
source would not have caught this. It does not reach the crop upstream trains at for the
memory ladder: 76 tokens is a fifth of 384, and N = 48 exhausts the card at 76. It does not
prove the other 20 `ttnn.embedding` sites in the other five models route, only that the verb
no longer stops them. And it is not a statement about training STABILITY: this campaign proves
the update rule over N steps and says nothing about drift over a 100k-step full run, a bound
that must be stated and not erased.
