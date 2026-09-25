<!-- Snapshot of ~/.coworker/state/of3t-gpugap.md, which is gitignored and has no
     other backup. The live doc is the one the orchestrator reads; this is the copy
     that survives. Re-derive every number with perf/of3t_gpugap/partition.py. -->

# of3t-gpugap — one number for "OpenFold3 training vs GPU", with its axis

TASK TYPE: VERIFY/BENCHMARK (read-and-arithmetic) | PLAYBOOKS loaded: VERIFY/BENCHMARK +
ALWAYS-ON | memories read: `a-harness-step-is-not-the-loops-own-round`,
`a-headline-figure-must-carry-the-axis-it-depends-on`,
`ratio-must-name-how-its-denominator-arm-was-built`,
`a-ratio-is-not-a-defect-until-the-references-own-ratio-is-measured`,
`your-own-worktree-already-holds-the-answer-grep-it-before-spending-device-time`,
`perf-page-matched-batch-protocol-recurrence`, `a-digest-with-one-carrier-is-a-transcription`,
`effort-spent-is-not-evidence-of-share`, `pooled-ratio-moves-when-its-weights-move`.

Branch `wk/of3t-gpugap`, worktree `/home/moritz/.coworker/wt/of3t-gpugap` on pc. No card, no
device run, no GPU rental. Every figure below is read out of an artifact already in git or
computed from one; the sources are named per row so each is re-checkable without a card.

VERDICT: GO — the headline is **58–67x at crop 384, whole training step against whole training
step**, and it is a floor. Two of the three ratios the brief carried are struck: one is a
different model, one is a TT-vs-TT number with no GPU on either side. The partition inverts BC2's
lesson rather than repeating it — OF3T's unported fraction is **1.0–1.9 %** of a step, not 79.2 %,
so at the measured configuration nothing caps a device-side lever below **~74x**. But that holds
only at the 4 diffusion samples the step was measured at. At their shipped **48**, the host numpy
loss heads alone project to **22–38 s per step** against the H200's entire 7–8 s step, so the
unported fraction goes from harmless to **decisive**: with the whole device side at zero cost the
step would still be 3–6x slower than the GPU. That is the lever list's first entry and the record
did not contain it. And the gap is **not a hardware gap**: upstream's own step at the same crop,
batch, precision and 48 samples takes **394.61 s on an 8-core desktop CPU**, so our Blackhole
step is 1.18x slower than a Ryzen while doing a twelfth of the diffusion work. A sprint that
reads 58–67x as a silicon problem is reading it wrong.

## MEASURED

AXES: every ratio in the record, with what each side's step actually contains.

| # | ratio | TT side | reference side | crop | verdict |
|---|---|---|---|---|---|
| 1 | **~5–6x** — 61–70 s/step vs ~12 s/step per rank (`of3t-perf.md:41-42`) | **Protenix v2**, not OpenFold3 | ByteDance's, per rank | unstated | **STRUCK — wrong model.** It sits under the literal sentence *"Calibration from the sibling campaign"*, beside ABodyBuilder3's 3.31x-vs-A100. Neither arm is OF3. `of3t-perf`'s own text four lines above says *"no ratio against our side is claimable yet, and none is given here"* |
| 2 | **23.7x** — 230.11 s taped vs 9.721 s untaped (`of3t-tapediverge.md:27-28`) | one p300c, trunk, taped, AICLK DURING | **there is no GPU side** — the denominator is the *same p300c* untaped | 384 | **STRUCK as a GPU ratio.** It prices what taping costs us, which is a real and useful TT-internal number, superseded at step scope by `of3t-stepfloor`'s **94.3x** (585.062 s taped vs 6.205 s untaped, same harness, same process shape, same card, same 4-sample config). It answers a different question |
| 3 | **46–53x** — 370.85 s vs 7–8 s (`of3t/EVIDENCE.md:89`) | p300c qb2 card 3, **trunk only**: fwd 13.53 + bwd 357.32, 2,473 tape nodes, probe off, AICLK median **1350** over 78 DURING samples, loadavg [10.16, 9.60, 9.72] | H200 1980 MHz, **complete Lightning step** — trunk + diffusion + every loss head + optimizer — bf16-mixed, batch 1 | 384 both | **VALID, SUPERSEDED.** The only matched-crop pair in the brief, and its own row says the axes are mismatched in the flattering direction: no diffusion, no losses, no optimizer on the TT side. A matched-scope replacement already exists |
| 4 | **58–67x** — **466.70 s vs 7–8 s** (`of3t-stepfloor.md:142-152`, `perf/of3t_stepfloor/out/step_rekey_b_384.json` rep 2) | p300c qb2 **dev3**, **full taped training step**: trunk + diffusion + loss heads + backward + optimizer, 9,888 tape nodes, 2,660 of 3,152 weights carrying a gradient, JIT-warm rep 2 of 4, quiet host (loadavg 9.1→12.2), AICLK median **1350** over 578 DURING samples | same H200 arm as row 3 | 384 both | **HEADLINE** |
| 5 | **1.18x** — 466.70 s vs **394.61 s** (`of3t-theirtest.md:68`) | the same 466.70 s full taped step as row 4 | **not a GPU** — upstream's own Lightning step on qb2's host CPU, AMD Ryzen 7 9700X, 8 cores, torch 2.8.0+cpu, `no_samples` 48 | 384 both | **CALIBRATION, and it is the one the table was missing.** Matched in crop, batch and precision, and their side runs 12x our diffusion samples. We are slower than eight Zen cores. See the section below |

Withdrawn by the record before this row, listed so nobody revives them: **870.75 s ⇒ ~116x**
(D164 — the allocator probe was 499.90 s, 57.4 % of that wall clock) and **222.48 s ⇒ ~28–32x**
(quoted a different harness's steady arm for this scope).

The GPU arm, stated once with its own weakness: epoch-0 per-step wall clock **11 / 8 / 7 s**,
first step warmup, `token_budget: 384`, batch 1, one device, bf16-mixed, H200 at 1980 MHz. That
is **n=2 steady samples** against the TT side's eight reps, and `EVIDENCE.md` records that §4a is
not satisfied — the recycle count is drawn U{0..3} and was not pinned, so 8 and 7 are a sample
over the draw. It is the thinner of the two measurements and the headline inherits that.

HEADLINE: **OpenFold3 training on one Blackhole p300c is 58–67x slower than one H200, comparing a
whole training step to a whole training step at crop 384, batch 1, bf16-ish against bf16-mixed —
466.70 s against 7–8 s — and the TT side is doing strictly less work, so 58–67x is a floor.**

It beats row 3 because it is the same crop *and* the same scope. Row 3's TT arm stops at the
trunk while its GPU arm is a complete step; row 4 puts loss heads, the diffusion module and the
optimizer on both sides. Everything else in the brief is off-axis: row 1 is Protenix, row 2 has no
GPU in it.

What still flatters the TT side, each with its size:
* **4 diffusion samples against their 48.** The largest unpriced term, and its provenance
  is worth stating exactly because the obvious citation is the wrong one. The GPU arm did **not**
  run the shipped `initial_training` stage yaml — it ran the runner yaml upstream's
  `test_training_full.py` generates (`scripts/datasets/pdb_subset_helpers.py:520`
  `build_runner_yaml_config`), which overrides `token_budget: 384`, `batch_size: 1`,
  `precision: bf16-mixed`, `epoch_len: 4`, `max_epochs: 2` and
  `loss_module.diffusion.chunk_size: 2`, and **does not touch `no_samples`**. The value falls
  through to the architecture default, read directly from upstream's source:
  `openfold3/projects/of3_all_atom/config/model_config.py:181`
  `architecture.shared.diffusion.no_samples: 48`, with `num_recycles: 3` at `:177`. The `train`
  preset that yaml selects is a **memory** preset — chunk sizes and kernel flags — and changes
  neither. The stage yaml lands on 48 too; this route is the one that is checkable, and `perf/of3t_gpugap/partition.py --upstream` asserts it against the source rather than quoting it. One bound on that re-check: the tree on pc is **0.4.6.dev12**, while `of3t-theirtest` ran **0.5.0**. So it proves the default is 48 on 0.4.6 as it is on 0.4.3 (`their_step_shape_0.4.3.json`) — 0.5.0 sits between two revisions that agree and is not re-read directly. Forward is cheap — marginal 0.043 s/sample, so 48
  samples is ~2.18 s of forward. The *backward* at 48 is **not projectable from this record**:
  2 samples read 642.54 s and 4 samples read 585.062 s, the wrong way round, because the
  cross-process cold A/A is 24.2 s and the quiet-vs-loaded spread is 18 %. Naming that as
  unmeasured is the honest answer; a sample-count ladder on a quiet host is what settles it.
* **`trunk_nograd_prefix_s: 0.0` on all eight reps** — `cycles_pinned: 1`, no no_grad prefix at
  all, while their draw is U{0..3}. At the 2.40 s untaped cycle (`of3t-perf`) a mean draw of 1.5
  is **3.6 s, 0.77 % of the step**. Small, and in the flattering direction.
* The fixture's ground truth is the model's own prediction plus noise, so `value_claimed: false`.
  The terms fire at the right count and the right shape, which is what a cost reading needs; no
  loss value and no gradient direction is claimed from it.

Flattering the GPU: the TT host was contended throughout (loadavg 9.1–12.2 even on the quiet arm),
and arm B rep 2 is the favourable rep. Using arm A's loaded reps instead gives 507–768 s.

## The calibration the ratio table was missing: upstream's own step on a desktop CPU

`of3t-theirtest.md:68` ran upstream's `test_training_full.py` on **qb2's host CPU** — AMD Ryzen 7
9700X, 8 cores / 16 threads, torch 2.8.0+cpu, no card opened — and their own Lightning loop
completed **one training step at crop 384, batch 1, bf16-mixed in 394.61 s**. Same generated
runner yaml as the H200 arm, so the same `no_samples: 48` and the same U{0..3} recycle draw.

**Our p300c full taped step is 466.70 s. Upstream's own step on an 8-core desktop CPU is
394.61 s. We are 1.18x SLOWER than the CPU while running 4 diffusion samples against its 48.**

Every caveat points the same way, which is what makes it usable:
* The CPU figure is the **first** step their loop completed, n=1, so it carries warmup. On the
  H200 the first step was 11 s against a 7–8 s steady state, 1.4–1.6x. A steady CPU step is
  therefore likely well under 394.61 s and the real gap larger. Direction, not a number.
* That CPU arm had **four autocast islands disabled** (`of3t-theirtest`), so parts of it ran
  wider than bf16 — which makes it slower, not faster.
* Shim 3 changed the model configuration, which is why that run is not reported as a pass of
  upstream's test. It disables the Triton triangle kernels on the **eval** path; the training
  step already has `use_triton_triangle_kernels: False` under the `train` memory preset
  (`model_config.py:97-104`), so the timed step is the unmodified one.

**This is the second, independent proof that the gap is an implementation gap and not a hardware
gap.** The partition below reaches it from inside — the backward's verbs cost 44.3x what the same
process's forward verbs cost. This reaches it from outside: a Blackhole losing to eight Zen cores
is not a FLOPs story. Both say the same thing, and a sprint that reads 58–67x as "we need more
silicon" is reading it wrong.

## PROVED — the partition

PARTITION: the headline step splits **98.6 % ported / 1.4 % unported**, and the split is measured
per part by timers that each end in `ttnn.synchronize_device(dev)` (`perf/of3t_stepfloor/
fullstep.py:391-450`), not inferred. Arm B rep 2, the headline step; the six parts sum to
466.702 s against the recorded `step_s` of 466.702 s exactly.

| part | s | share | ported? | evidence it is where I say it is |
|---|---|---|---|---|
| trunk, one taped cycle | 3.415 | 0.73 % | **PORTED** | ttnn verbs, sync-bracketed |
| diffusion, 4 differentiated samples | 0.287 | 0.06 % | **PORTED** | ttnn, `coupled_to_trunk: true` |
| loss heads, `af3_loss` | 3.135 | 0.67 % | **UNPORTED** | `tt_bio/train/losses.py` and `objectives.py` contain **0 occurrences of `ttnn`** — pure numpy. The PCIe leg is 0.006 s of the 3.135; the other 3.128 is host arithmetic |
| seed upload | 0.003 | 0.00 % | seam | |
| backward, trunk + diffusion | 456.668 | **97.85 %** | **PORTED** | `tt_bio/autograd.py`, 210 ttnn call sites |
| optimizer, AdamW over 3,152 | 3.194 | 0.68 % | **UNPORTED** | `tt_bio/train/optim.py:153-165` — fp32 masters and both moments are numpy on the host. Its own docstring: *"`ttnn.moreh_adamw` takes `param_in` as bfloat16 or bfloat8_b only, so the device cannot express an fp32 master at all and the placement is forced rather than picked"* |
| **STEP** | **466.702** | | | |

Across all eight reps in `step_rekey_384.json` + `step_rekey_b_384.json`: backward **97.33 % to
98.30 %**, unported **1.04 % to 1.94 %**. So the partition is a property of the configuration, not
of one rep.

**This is the opposite of BC2.** There, 79.2 % of the round was host work no lever could reach and
every credited lever capped at 1.25x. Here the host fraction is 1.0–1.9 % and essentially the
entire step is inside code we own.

**The second-level partition, which is what actually ranks levers.** Within the backward, how much
is the model's arithmetic? The same program, forward and backward, one process, one card
(`of3t-stepfloor.md:28-31`, D164 pass 2, JIT-warm):

    arm                 s        ttnn verb calls    ms per call
    forward, warm       4.04              84,996        0.0475
    backward          356.00             168,922        2.1075     44.3x per call

The backward issues **1.987** verb calls per forward call, which is what reverse mode costs, and
each acts on operands of the same shapes. **At the forward's own measured warm rate those 168,922
calls would cost 8.03 s. They cost 356.00 s.** Even pricing a backward verb at 5x a forward verb
leaves 40.1 s predicted against 356.00 s measured — still **8.9x short**. So **~348 s, 97.7 % of
the backward and ~75 % of the whole step, is not the model's arithmetic**, and it is not JIT
either: this is rep 2 of a warm process, and the JIT term was separately measured as the forward's
13.53 → 4.04 s (`of3t-stepfloor`), which is already burned off here.

Scope of that second-level reading, stated rather than glossed: the verb counts exist only at
**trunk** scope (`ladder.py`), because `fullstep.py` records no per-part call counts. The full
step's backward (456.67 s) is the same object one harness out and agrees in order. The one named
host exception inside the tape is `autograd.py:1796`, a numpy `cdist` VJP; whether it fires on
this step is unmeasured.

Mechanism, from the code rather than from the number: `_backward` (`tt_bio/autograd.py:562`)
is a **python reverse-topological walk firing one closure per node**, with a per-node `_unshard`
and a conditional `ttnn.typecast`, dispatching every verb individually. There is **no trace
capture on it**, while `tt_bio/esmc.py:1063`, `tt_bio/protenix.py:1208` and
`tt_bio/tenstorrent.py:13633` all already use `ttnn.begin_trace_capture`/`execute_trace`. The tape
is shape-static across steps — 9,888 nodes and `rebind_moved: 3152` on every rep — which is
exactly the precondition trace capture needs.

CEILING: the unported fraction's arithmetic, and it has two answers depending on the sample count.

* **At the measured 4-sample configuration**: unported is 6.329 s of 466.702 s, so a device side
  reduced to zero leaves 6.329 s, a **73.7x** best case (51.4x to 95.9x across the eight reps).
  The headline gap is 58–67x. **The unported fraction does not block parity here — it permits it
  with about 10 % to spare.**
* **At their shipped 48 samples, it does block it.** `host_losses` loops over roots
  (`fullstep.py:253`, one `af3_loss` call per root) and roots = diffusion samples, so the host loss is linear in sample count:
  0.455–1.226 s per root across the eight reps ⇒ **21.8 s to 58.8 s per step at 48 samples**
  (quiet arm B: 21.8–37.5 s). Add AdamW's 3.0–3.2 s and the unported floor is **24.8–40.7 s**.
  Against the H200's **entire** 7–8 s step that is **3.1x to 5.8x slower with every device op
  costing nothing**. Parity at their real recipe is arithmetically impossible until `af3_loss`
  stops being host numpy, and no measurement in the record showed this because every step was
  taken at 2 or 4 samples.

LEVERS: ranked by the share of the step each can close, with the arithmetic. Nothing here is
landed; this is the ranking the partition supports.

1. **The backward's per-verb cost — 456.67 s, 97.85 % of the step.** The only lever that can move
   the headline. Target: the ~348 s that is not arithmetic. Bring the backward from 2.108 to
   0.19 ms/call (4x the forward's warm rate, a deliberately unambitious target) and it costs
   ~32 s, putting the step at ~42 s and the gap at **5.6x**. **The sub-lever cannot be ranked from
   this record**, because nothing separates host dispatch from kernel inefficiency from extra ops.
   *The sprint's first purchase should be one device profile of a steady taped backward at crop
   384 on a quiet qb2* — a card-hour, not GPU credit. The leading candidate is trace capture on
   `_backward`: the tape is shape-static, the mechanism is already in tt-bio three times, and a
   python per-closure dispatch loop is the shape of cost that trace removes entirely.
2. **Port `af3_loss` off host numpy — 0.67 % today, 22–38 s/step at their 48 samples.** Ranked
   second only because it is invisible at the configuration everything was measured at. It is the
   *binding* constraint on ever reaching parity (see CEILING), and it is worth nothing at all until
   lever 1 lands, which is why it must not be done first. Cheapest intermediate step: vectorise the
   root loop over the sample axis instead of porting, which is a host change with no accuracy risk.
3. **AdamW's host round trip — 3.0–8.0 s/step, 0.7–1.9 %.** Not a port: `ttnn.moreh_adamw` cannot
   express an fp32 master, so this needs either a hand-written fp32 Adam kernel or an upstream ask.
   3.2 s is 43 % of the GPU's whole step, so it matters after 1 and 2 and not before. Its spread
   (2.99 to 8.00 s on identical work) is host load, which is also the cheapest thing to fix.
4. **The forwards — 3.70 s combined, 0.79 %. Deprioritise, explicitly.** Zeroing the trunk cycle
   *and* the diffusion forward moves the step by under 1 %. This refutes where the campaign's perf
   attention has gone: `of3t-perf` measured the forward exhaustively (2.40 s/cycle to 0.3 % across
   processes, a rollout ladder, three instrument defects closed) and the backward not at all, and
   the backward is 98 % of the step.

What the ceiling permits overall: **at 4 samples, 73.7x, so the 58–67x target is reachable on
levers 1 and 3 alone.** At 48 samples nothing reaches parity without lever 2, whatever lever 1
achieves.

SPEND: none. Zero vast.ai hours, zero instances started, zero GPU credit. No GPU measurement was
needed and none is requested: the matched-crop H200 figure already exists at `EVIDENCE.md:89` and
this row is arithmetic over artifacts already in git. If the sprint wants the GPU side strengthened
from n=2 steady samples to a pinned-recycle median, that is a separate ask and it belongs behind
the device profile in lever 1, which costs a qb2 card-hour and settles far more.

## What would change the headline

One measurement, and it is not a GPU one: **a diffusion-sample ladder (4 / 12 / 24 / 48) of the
full taped step on a quiet qb2**. It converts 58–67x from a floor into a figure, and it prices
lever 2 against lever 1 instead of leaving their order argued. The eight reps here cannot do it —
the 18 % host-load spread swamps the sample-count effect, which is why 2 samples read slower than
4. A quiet host is the whole requirement.
