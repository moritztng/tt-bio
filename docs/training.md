# Training: the four tiers, and what each one guarantees

Progressive disclosure, cut where the **unit of user authorship** changes. Not where the
amount of configuration changes, which is the distinction this whole design turns on: you can
name your authorship unit before you start, while "how much configuration" is only knowable
after you hit a wall.

| Tier | Surface | You own | Cut line, and its test |
|---|---|---|---|
| 0 | `tt-bio finetune ...` | a config | no callables in the signature |
| 1 | `train.finetune(...) -> Run` | the objective | no `for` over steps in your code |
| 2 | `plan`, `batches`, `objectives`, `AdamW`, `Checkpointer`, `Mesh`, `LoraConfig`, `trainable`, `attach` | the `for` statement | no `ttnn` call in your code |
| 3 | `tt_bio.autograd` + `train.gradcheck` | an op and its backward | `ttnn` appears here |

Each cut line has a test in `tests/test_train_interface.py` that can fail. A boundary that is
only described drifts, and three of the four tests below caught a real defect while being
written: the Tier-0 dry run was importing the tape, `from tt_bio.train import plan` was handing
back a module where the tier promised a callable, and the naming collision was flagged by a
gate rather than by taste.

## The three calls, and what changes between them

The design is against real usage, so these are the three a user actually writes. They differ by
one argument each, and none of them is a rewrite of the one before it.

```bash
tt-bio finetune data/ --model protenix-v2 --out runs/a --global-batch 8 --steps 2000
tt-bio finetune data/ --model protenix-v2 --out runs/b --global-batch 8 --steps 2000 --chips 0,2
tt-bio finetune data/ --model protenix-v2 --out runs/c --global-batch 8 --steps 200000 --train weights
```

```python
from tt_bio import train

run = train.finetune(forward, dataset, out_dir="runs/a", global_batch=8, steps=2000)
run = train.finetune(forward, dataset, out_dir="runs/b", global_batch=8, steps=2000,
                     mesh=train.Mesh({"dp": [0, 2]}))
run = train.finetune(forward, dataset, out_dir="runs/c", global_batch=8, steps=200_000,
                     train="weights")
```

**Multi-card is `--chips`, and nothing else moves.** The recipe is one process; it sees a wide
axis and hands the run to the launcher, which re-runs your program once per chip. The loop is
the loop it was on one chip.

**Pre-training is `--train weights`, and nothing else moves either.** `adapters` puts a LoRA
factor pair beside each site and freezes the trunk; `weights` trains the model's own weights at
those same sites. The loop, the objective, the optimizer, the checkpointer and the data-parallel
axis are the same code, and the escape-hatch test below compares them instruction for
instruction rather than taking the claim on trust. What does change is the memory arithmetic, so
`--train weights` is a name a flag can carry and `plan()` answers differently for it.

## Why four, and why here

Three would leave too wide a gap. Going from "fine-tune with LoRA" to "write ops on a tape" is
weeks of work, not an afternoon: costing a per-module backward by hand
(`perf/hall_grad/DECISION.md`) puts it at 3 to 6 engineer-days per module. Tier 2 is the step
in between, and it exists so nobody has to take the whole drop at once.

The boundaries come from what the surveyed frameworks got wrong rather than from taste.
Hugging Face cuts at quantity of configuration, so its boundary lands in the middle of the
loop and its escape hatch has to hand an invariant back to the user. Lightning cuts at
inversion of control, so crossing costs the entire loop in one go. `accelerate` silently
redefines global batch and scheduler cadence in terms of device count
(`data_loader.py:347-348`, `scheduler.py:69-76`). That is exactly the axis a published recipe
pins, so the same script trains a different recipe on a different machine and nothing says so.

## The escape hatch is a mechanism, not a promise

Everyone ships a hatch. Almost nobody ships one that still works a year later, because the tier
above quietly grows a dependency on something the tier below cannot reach. The durable form:
**an escape hatch is only real if the tier above is expressible in the tier below's public API,
and it only stays real if a test asserts it.**

So the arrangement is deliberately rigid:

- `train.finetune` implements no loop. It looks a recipe up in `tt_bio/train/recipes.py` and
  calls it. There is one code path, so "the hatch reproduces the Tier-1 result" is identity
  rather than two implementations agreeing today.
- `train.recipes.source(name)` returns that same function's own text, via `inspect.getsource`.
- `tt_bio.train.TIER2` is the complete vocabulary a recipe body may use. The test compiles the
  body and reads its `LOAD_GLOBAL` instructions, which is what CPython will actually look up in
  globals, so the check is exact rather than an AST approximation of scoping. Today the recipe
  reaches 22 globals: 16 Tier-2 names and 6 builtins, zero leaks.
- A deliberately broken body that reaches a private helper fails the same check. The gate is
  known to be able to fail, which is the only reason to trust it when it passes.
- A second arm execs the returned text in a namespace holding nothing but `TIER2` and compares
  the result to the shipped recipe instruction by instruction. It runs where the wheel is
  installed.

If a recipe ever needs something private, that test fails and the name becomes public (a
deliberate widening of `TIER2`) or the recipe changes. There is no third option.

Note on comparing compiled code: raw `co_code` equality is the wrong assertion. Two
compilations of the same text legitimately differ, because a nested code object embeds its
filename, jump targets shift with it, and CPython picks between `LOAD_METHOD` and
`LOAD_GLOBAL`+`LOAD_ATTR` depending on how the enclosing code was compiled. Measured on this
recipe: 383 instructions either way, identical names and identical nested constants, three
positional differences and one such pair. The test compares the normalised program, so it
cannot pass on a changed recipe and cannot fail on a compiler detail.

## Invariants that live in the API

Each is a bug that happened, so each is enforced where it is cheap to act on rather than
written down where it can be skipped.

**`gradcheck` keeps three levels of evidence, in the order that makes each trustworthy.** The
float64 reference is checked against central finite differences first, in float64, never
touching a device. A reference nobody verified is how a campaign spends weeks chasing a
confident wrong number. Then the device gradient is compared to that verified reference, with
inputs rounded to the device dtype first so what is measured is the op's error and not the
input quantisation. Then the controls: an fp32 arm must show the error collapse, and a
deliberately broken arm must fail. A reference that fails level 1 reports itself rather than
the device, because sending the next reader to the wrong file is worse than no result.

Level 1 runs anywhere, including a box with no card:

```
pytest tests/test_autograd_reference_gate.py
```

Twenty-six references, one second. It checks what the device is scored against, not the
device. `--seed 7` reproduces a run exactly, in any process.

**The bars are per op class, off measured floors.** One bar for every op is wrong twice over.

| class | fp32 | bf16 | where the bar comes from |
|---|---|---|---|
| eltwise | 3.0e-06 | 1.0e-02 | 10x the measured fp32 eltwise floor 3.0e-07 |
| reduction-carrying | 2.5e-02 | 1.0e-02 | 3.5x the measured fp32 HiFi2 matmul floor 7.05e-03 |
| kinked | not scored | not scored | direction and gate-mask agreement instead |

A matmul on fp32 operands keeps about 11 mantissa bits whatever the kernel config says, and
reductions round like matmuls, so an fp32 arm does **not** collapse a reduction-carrying
gradient to machine epsilon, and scoring one against an eltwise bar fails a correct op. And a
mantissa-derived bar assumes the gradient is continuous in its input, which it is not at a
kink. `linear`, `linear_nobias`, `linear_silu`, `linear_sigmoid` and `layernorm` land at
0.0024 to 0.0049 against a 0.01 bar while `linear_relu` fails at 0.0625, and raising fidelity
to HiFi4 only reaches 0.0128. So a kinked op is scored on direction and on the fraction of gate
coordinates it agrees on. What it must not get is a smooth-op bar plus an excuse.

**The optimizer refuses a bf16 master, at construction.** Only 0.209 of an `eps=0.01` step
survived a bf16 round-trip while the gradient's own norm still read healthy. The failure has no
signature in the loss, the gradient norm or the step count, the weight simply stops moving, so
before the run starts is the only cheap place to catch it. The device copy the forward reads
stays bf16; that is the point of having a master, and it is written by casting the master down
each step.

**The step control is on the cumulative displacement ratio, never per-step.** With an fp32
master behind a bf16 device copy, a single step below bf16 spacing is *supposed* to round to
zero, so the per-step kept ratio scatters far from 1 on a run that is working perfectly. A
per-step assertion fails the healthy case. What has to hold is that the device weight has
travelled as far as the master has over the run: `opt.check_displacement()` asserts
`0.9 < ratio < 1.1` and raises, with the two displacements quoted, rather than warning.

**Every run carries provenance, and the clock is sampled during the work.** On this hardware
the clock sets the time. The 512 aa cell reads 21.90 s at 800 MHz, 17.34 s at about 1063 and
14.69 s in the 1350 burst, so a number recorded against a clock read before the work started
is not a measurement of the work. The sampler reads sysfs and `/proc/<pid>/fd` and opens
nothing, so it cannot perturb what it measures. It samples every card and separately records
which device nodes the process holds, because a sampler keyed to `TT_VISIBLE_DEVICES` can watch
a different chip than the one computing. Below 1200 MHz the record says the timing is a
governor artifact instead of reporting a regression. Accuracy is recorded as a deviation
against the 0.60 A bar with the 1.84 A seed floor beside it, never as a bare "passed".

## `plan()`: what is measured, and where it says UNMEASURED

`plan()` answers "will this fit and how long" and will not dress a projection as a number.
`UNMEASURED` is a first-class answer.

Measured, and each carries its source:

- **34.23 GB** of card DRAM, read live off a Blackhole chip as 8 banks x 4,278,190,016 B.
- A Protenix-v2 training replica at 256 aa is **5.06 GB, 14.8 %**: 0.929 GB bf16 weights over
  464,442,431 censused parameters, 0.429 GB fp32 trunk gradients, 3.70 GB activations at the
  backward sync point under per-block checkpointing. With optimizer state resident it would be
  **10.63 GB, 31.1 %**. It is not, because masters and both Adam moments are host-side.
- **1.87x on two chips, 93.5 % efficiency**: 8.08 tokens/s on one chip, 15.11 on two, 1350 MHz
  sampled during on both.

Refused rather than estimated:

- **384 aa**: 4.14 GB allocated, 75,497,472 B refused. **512 aa**: 7.15 GB, 536,870,912 B
  refused. Both measured, both in the forward under per-block checkpointing where the forward
  is untaped, so what fails is one block's working set. Distribution does not help: 8 chips
  each run out at 384 aa exactly as one does. A crop re-measure on the shipped forward is
  scheduled, and until it lands `plan()` refuses instead of extrapolating a slope through two
  failures.

`UNMEASURED`, with the reason:

- **Above 256 aa**, and not one of the two measured failure sizes. Activation volume is neither
  linear nor quadratic in tokens across the triangle operations' chunking thresholds, so
  interpolating between 256 and 384 would be a guess with a plausible shape.
- **Anything above a LoRA adapter on a frozen trunk.** The only source for a trained trunk's
  memory is `perf/hall_grad/DECISION.md`, a feasibility memo whose 27.58 GB and roughly 40
  engineer-days are projections of work that is not built, and which says so itself.
- **A step time on more than two chips.** 1.87x at two chips is the only multi-chip point we
  have, and carrying 93.5 % forward as a per-chip efficiency assumes the host reduce's per-rank
  volume and shard imbalance stay flat in rank count, which is exactly what is unmeasured. Two
  points cannot measure a scaling exponent.

Below 256 aa `plan()` reports the 256 aa figure as an **upper bound** rather than scaling it
down: a smaller crop carries the same weights and optimizer state and strictly fewer
activations, so the bound is sound and the scaling would be the guess.

## Distribution

Single box, up to 4 chips, data-parallel. In user code it appears exactly twice:

```python
mesh = train.Mesh({"dp": [0, 1]})
opt = train.AdamW(params, lr=3e-4, data_parallel=mesh.axis("dp"))
```

**The all-reduce belongs to `step()`, not to a callback.** tt-train's own inconsistency is the
argument: `SFTTrainer` requires the user to remember a `DDPCallback` while `GRPOTrainer` does it
itself, and forgetting it trains N diverging replicas behind N loss curves that all look right,
because each replica's loss is real, it is just a different model's loss. No warning is
available at the callback layer, since a missing callback is indistinguishable from a user who
meant one chip. So `opt.step()` raises `UnreducedGradients` when the axis is wider than one
chip and no per-chip gradients were passed. Sum, not mean: the divisor is the global batch you
pinned, and dividing by chip count here is the substitution that turns one recipe into a
different one per box.

### The launcher: one process per chip

A wide axis needs one process per chip, and `tt_bio/train/launcher.py` is what makes one. Ask
for it and it happens: `tt-bio finetune ... --chips 2`, or `train.finetune(..., mesh=Mesh({"dp":
[0, 1]}))`. The Tier-1 recipe is a single process, so when it sees a wide axis and nothing
launched it as a rank, it hands the run to `launcher.drive`.

**It re-runs your program, it does not ship objects to it.** `TT_VISIBLE_DEVICES` is read at
`import ttnn`, so a fork cannot give a child a different chip than its parent, and an unpinned
open brings up every visible chip rather than the one being computed on. A rank therefore has to
be a fresh interpreter, and a fresh interpreter cannot be handed a live forward or a dataset
holding a device. So the driver re-executes your command line with `TT_BIO_DP_RANK` set, like
`torchrun`, and each rank builds its own forward and dataset. Two things follow, and both are
checked before anything is spawned rather than documented and hoped for:

- **Your program must be re-runnable** up to the `finetune` call. `python -c` and an interactive
  session are refused by name, because there is no program to re-run.
- **It must hold no card when it gets there.** Resolve your dataset's `device` lazily, on first
  use, so the open happens inside the rank that computes on it. A driver that already has
  `/dev/tenstorrent/N` open is refused with that sentence.

**The gradients are summed on the host, through `/dev/shm`, deliberately not on the device.**
The masters and both Adam moments already live on the host because `ttnn.moreh_adamw` cannot
hold an fp32 master, so the gradient crosses PCIe to reach them whatever happens; an on-device
collective would move it to the device and back for nothing. The whole parameter set goes in one
message per step, because across processes the cost is the barrier and not the bytes.

**Two failures look exactly like a healthy run, so both are refused rather than reported.**

- *The ranks diverged.* They start from one seed and a reduce before every step keeps them
  identical, so the launcher hashes every rank's masters at the end and requires exactly one
  distinct hash. This is not hypothetical: it caught `lora_factors` seeding its adapter init
  from entropy, which had every rank starting from a different `A`. Both loss curves fell and
  the step times were normal.
- *Two ranks shared a chip.* `TT_VISIBLE_DEVICES=N` does not open `/dev/tenstorrent/N`; on a
  QuietBox the UMD ids run in PCI-BDF order and the node numbers do not, so `1` lands on node 2.
  Two ranks on one chip with a third idle still finish, still agree on their masters, and still
  show a speedup. So each rank's node is read from that rank's own `/proc/<pid>/fd` after the
  device is up, and the set has to be distinct and of size `world`.

Measured on qb1, two p150a chips, both arms at 1350 MHz sampled during the run from inside the
measuring process, cotenants on the other two chips:

| adapter gradient | 1 chip | 2 chips | speedup | efficiency | exchange | barrier |
|---|---|---|---|---|---|---|
| 0.33 MB | 0.2578 s/step | 0.2636 s/step | 1.956x | 97.8 % | 1.01 ms | 1.09 ms |
| 5.24 MB | 0.2793 s/step | 0.3288 s/step | 1.699x | 84.9 % | 7.48 ms | 4.96 ms |

The exchange is 2.3 % of the step at 5.24 MB, so it is not what costs the 15 %: the host-side
Adam is. It runs on the CPU because the optimizer keeps an fp32 master, its cost scales with the
adapter size (+21 ms per step from the 16x larger adapter on one chip alone), and two ranks on
one host contend for it. That is the thing to attack if DP efficiency ever needs to be higher,
not the collective. `perf/train_d_dp/` holds the harness and the raw results.

Tensor parallelism is deliberately absent. A full replica is 14.8 % of one chip, so there is no
memory argument for it, and it returns only if something later forces it.

Multi-host is out of scope, and the blocker is cabling rather than software: 20 MB/s over WiFi
makes a per-step gradient exchange cost more than the step. `Mesh.auto()` reports one host
today, and the interface does not change shape when that changes: an axis is an axis whether
its chips share a host or not, and going multi-host is meant to be the same one argument that
going multi-card is. Until it is wired, the honest claim is multi-card on one host.

## Featurisation is per model, on purpose

The one thing this design refuses to generalise, and the biggest trap in it rather than the
biggest win. A shared data layer would have to model every family's cropping and MSA handling,
and each family's is exactly the part that is genuinely different. So
`tt_bio/train/catalogue.py` ships empty and a model registers its own:

```python
from tt_bio.train import catalogue
catalogue.register("mymodel", lambda path, tokens=None: (forward, MyDataset(path)))
```

A dataset needs four members (`__len__`, `tokens`, `device`, `batch(indices) -> dict`) and no
base class. Until a model registers one, `tt-bio finetune --model X` refuses with the name of
what is missing, which is the featuriser and not the interface.

## Adapting a model: how the weights and the sites are found

`trainable()` is the one call the loop makes, and what it hands back is the whole of the
adapters-versus-weights choice: a LoRA config gets `{site: (A, B)}` factors, `None` gets the
model's own weights. Either way it is one discovery forward and then `attach`, which is why the
loop above does not branch on which mode it is in.

**Pass `model=` the built model.** With it, `--train weights` discovers by WALKING the model, and
that is the only form that reaches every weight: a module that fuses two projections in its own
`__init__`, or pushes a fused weight on its first call, holds a tensor the loader never produced
and no routed call ever passes. Measured on a 4-block pair stack at 64 tokens, and the same
numbers on OpenFold3, Boltz-2 and BoltzGen: 156 weights visible at the loader, 188 reachable from
the built model, 204 after one forward. Without `model=` discovery falls back to the call-site
census below, which is what LoRA uses and what a Tier-2 caller holding a forward and no model
still gets.

Two things follow from walking, and both are checked rather than assumed.

* **The discovery forward runs first, and it runs taped.** Weights a module fuses lazily do not
  exist until the first call at a given shape, so a walk of a freshly built model is a walk of a
  smaller model than the one that runs. And several fused kernels decline while a tape is open,
  which routes the call down a composed path that fuses a different weight again — an untaped
  discovery forward reached 196 of the 204 weights the taped forward then used. `walked_weights`
  spends one `no_grad` forward with the hook installed, which costs the same one inference the
  census costs.
* **`params.rebind()` after every `step()`.** `AdamW.step` replaces a parameter's device tensor
  rather than writing into it, so the model's own attribute still holds the tensor discovery saw.
  `rebind()` puts the new one back where the walk found it. Skip it and the gradients are real,
  the loss curve falls and the model stands still. The loop in `recipes.py` calls it; a weight
  held somewhere unwritable raises by name instead of being skipped.

`train.checks.weight_coverage(model)` reports what a run reaches as leaves against total and
names every miss, because a leaf count with no denominator cannot show a shortfall.

`lora_factors_for` runs the shipped forward once with a census hook installed. Every call that
routes through `tt_bio.ops.linear` announces itself with its own shapes; a call that does not
route through it is not adaptable, and the census says so by not listing it. The hook records
and then declines every call, so a census pass computes exactly what an inference pass computes
and the output is comparable byte for byte.

Sites are named by call site (`file:line:qualname`) rather than by weight identity, because the
same weight is read at one site while two different weights are read at one shared helper. The
call site is the thing you can find and target with `--target`.

**A weight cannot be named that way, and the difference is correctness rather than taste.** A
48-block trunk whose blocks run the same line reports one site carrying 48 different weights.
One adapter that every block reads is a legitimate model; one weight that every block reads is a
different model from the one on disk. So `--train weights` names a parameter per
`(site, weight)`, numbered by first appearance: `site`, `site#1`, `site#2`. The numbering is the
same on every data-parallel rank, because every rank walks the same blocks in the same order off
the same checkpoint.

Two failures of that mode are refused rather than reported, both because they look like healthy
runs. The forward is handed the **parameter** and never the weight it passed in: `AdamW.step`
replaces the parameter's device tensor while the model's own cache still holds the one it
uploaded, so a forward reading the passed weight would train perfectly and never move. And a
forward that uploads a fresh weight per call is refused by name, because the optimizer would own
tensors nothing reads twice. The obvious version of that second check does not work: parameter
names are a deterministic function of call order, so a re-uploading forward mints exactly the
same names. The identities discovery saw are carried into the run instead.

`attach` composes over whatever hook is already installed instead of replacing it. A frozen
trunk still has to stay taped downstream of its first adapter or the gradient never reaches the
adapters in the early layers, so declining a non-adapter site outright would train only
whatever sits after the last adapter, with no error and a loss curve that still falls.

## The learning-rate schedule: your first step runs at lr(0)

`AdamW` reads its schedule before it advances its step counter, so update k runs at `lr(k-1)`
and the first update runs at `lr(0)`. Under the AF3 warmup that is exactly zero, so step 1
moves nothing. This is not a quirk to work around: it is the order upstream trains in.
Lightning calls `optimizer.step()` and then `scheduler.step()`, and `AlphaFoldLRScheduler` is
constructed with `last_epoch=-1`, which steps it once to 0 before training starts.

It matters if you are comparing a run against a reference: read the rate off `opt.last_lr`
rather than computing `af3_lr(k)` yourself, or you will be one rung further along the warmup
than the run was. Checked against upstream's own scheduler driven on a real `torch.optim.Adam`
over 2,005 steps, exact at every one: `perf/of3t_updaterule/lr_wiring.py`.

## Labels a loss term uses only if you supply them

`batch(indices)` returns the labels the objective names, and for `mse` three of them are
optional in the API and not optional in the loss. AlphaFold 3 upweights DNA and RNA tokens by
5 and ligand tokens by 10, and `losses.mse` applies that weighting only through `is_dna`,
`is_rna` and `is_ligand`. A featuriser that does not emit them trains every nucleic-acid and
ligand token at protein weight, and the term still fires, so no loss value looks wrong.

You can supply the fact either way. OpenFold3's pipeline emits the three flags directly.
Protenix-v2, Boltz-2 and BoltzGen call the same fact `mol_type`, a single integer column, and
`af3_loss` derives the three flags from it, so those batches carry the weighting without a
featuriser change.

Two things to know when you write a dataset:

* **If you emit `mol_type`, say which convention it uses.** The stacks disagree. Set
  `batch["mol_type_convention"]` to `"af3"` (protein 0, rna 1, dna 2, ligand 3, which is
  OpenFold3's and Protenix-v2's) or `"boltz"` (protein 0, dna 1, rna 2, nonpolymer 3, which is
  Boltz-2's and BoltzGen's). Both put ligand at 3 and they swap dna and rna. Leave it out and
  the split is assumed, which is harmless while AlphaFold 3 weights dna and rna the same;
  `entity_flags` refuses rather than guessing if that ever stops being true.
* **Check the breakdown.** `af3_loss` lists absent labels under `breakdown["mse"]["without"]`
  and records a derived set under `breakdown["mse"]["derived"]`. `without` is the only
  difference between a batch with no ligand and a batch whose featuriser never mentioned one:
  the loss value and the gradient are identical in both cases.

On a 56-token OpenFold3 batch with two ligand tokens, supplying the three flags moves the loss
0.233 and the gradient it seeds 0.764. Deriving them from a `mol_type` column instead gives
the same loss and the same gradient, bit for bit.

## What the recipe pins, and where it differs from Adam's defaults

`finetune` reproduces OpenFold3's optimizer setup rather than the library defaults it would
otherwise inherit. Three of those differ, all three are arguments, and none of them shows up
in a loss curve or a gradient norm:

| argument | recipe | library default | why |
|---|---|---|---|
| `betas` | `(0.9, 0.95)` | `(0.9, 0.999)` | OpenFold3 sets `beta2: 0.95`. Adam's usual 0.999 is a second-moment horizon twenty times longer and it changes the size of every update. |
| `weight_decay` | `0.0` | `0.01` | OpenFold3 builds a plain `torch.optim.Adam`. Any decay moves every weight whose gradient is zero. |
| `plateau_until` | `50000` | `None` | Selects AlphaFold 2's schedule, which holds the rate flat before decaying. `None` selects Protenix's, which decays from step zero. |

Pass your own if you are training against a different recipe. `plateau_until=None` is the one
to reach for first: it is the whole difference between the two AlphaFold-family schedules, and
over 200,005 steps they disagree at exactly one of them.

**The loop runs one sample at a time.** Per-sample clipping needs each sample's own gradient,
so `finetune` does one forward and one backward per index in the batch and accumulates, rather
than one forward over the whole micro-batch. At `--global-batch 8` on one chip that is eight
forwards per step. It is the same arithmetic upstream does and the reason is below.

## Gradient clipping: two things to pass, or you train a different rule

`AdamW` clips on the global norm at `clip_norm=10.0`, which is upstream's own value. Two
things decide whether that is the same rule the reference applies. `finetune` does both for
you; at Tier 2 you own the loop, so you own them:

* **`disabled=` the parameter names this sample does not activate.** They are excluded from the
  global norm and from the update, which is what OpenFold3 does — and it is not a corner case.
  Their runner disables the confidence head whenever a sample's summed confidence weight is
  zero, which the initial-training config does on four of its five datasets. Norm over a set the
  reference excluded and the coefficient applied to every gradient in the step is different:
  measured at 8.368e-01 relative on one small enabled tensor beside one large disabled one.
* **`clip_and_accumulate()` after each sample's backward, instead of one clip per batch.**
  Per-sample clipping is a different algorithm and not a different constant: it bounds each
  sample's contribution, so it changes the direction of the accumulated update, not just its
  length. Measured at 1.948e-01 relative over three samples with one 200x outlier.
  `per_sample_clipping: True` at `clip_val 10.0` is upstream's shipped default. `step()` then
  divides each parameter's accumulated gradient by the number of samples that actually
  activated it, and does not clip again.

Both are verified against OpenFold3's own `grad_manager`, executed rather than transcribed:
`perf/of3t_leaves/clip_equiv.py`.

## The host float64 softmax: for when fp32 is not fp32

`TT_BIO_HOST_F64_SOFTMAX_AB` moves a softmax off the card and computes it on the host in
float64. It is off everywhere and it is a training lever: what it buys is gradient fidelity.

The reason it exists is that a Tenstorrent fp32 is a few mantissa bits short of an IEEE one, and
a softmax is where that shows. On `[1,16,384,384]` fp32, scored against a float64 softmax of the
same values on one Blackhole processor of a p300c at 1350 MHz:

| softmax | error vs float64 | ms/call |
| --- | --- | --- |
| the op's own default | 2.029e-02 | 0.0545 |
| `TT_BIO_SOFTMAX_PRECISE_AB` | 1.646e-03 | 0.0790 |
| `TT_BIO_ACCURATE_SOFTMAX_AB` | 5.156e-04 | 0.2576 |
| `TT_BIO_HOST_F64_SOFTMAX_AB` | 2.082e-08 | 9.0966 |

A real fp32 softmax lands around 1e-7, so the first three are all four or more orders of
magnitude away from it and no configuration closes that: it is the silicon. OpenFold3 trains in
IEEE fp32 on GPU, so the round trip is not overshooting them, it is the route to what they
already do.

What it is worth. On the OpenFold3 diffusion module's gradient over 547 parameters, 51.1358 % of
the model's squared gradient norm, against upstream's own bf16 training step, it takes the
mass-weighted relative error from 7.426217e+00 to 7.777580e-02 — 95.5x of the gap. Against an
exact float64 reference it reads 5.930664e-02, which is 1.013x what upstream's own bf16 step
reaches against the same reference. It is not free: 167x on the softmax alone, and 1.47x on the
whole gradient arm, because the softmax is a small share of what the step runs.

Syntax is the one the other per-site softmax flags use. A bare token turns one construction site
on, a `-` prefix turns it off, and `all` / `-all` move every site without a token of its own:

    TT_BIO_HOST_F64_SOFTMAX_AB=all                          # every site
    TT_BIO_HOST_F64_SOFTMAX_AB=openfold3.diffusion_transformer

The sites are `openfold3.diffusion_transformer`, `openfold3.atom_transformer` and
`protenix.atom_transformer`.

Predictions are untouched. With the flag unset, OpenFold3, Protenix-v2 and OpenDDE each write a
structure byte-identical to the one they wrote before this path existed, same card and same seed.

**You already have the cheap fix.** `TT_BIO_SOFTMAX_BW_RENORM` is on by default. It divides the
softmax backward's inner sum by the row sum, two extra ops and no change to any forward. It
exists because `d_logits = y(g - Σ g·y)` is only row-sum-free when the row sums to one, and
`ttnn.softmax` returns rows that miss it by up to 3.9e-02. On the same OpenFold3 gradient it reads 1.057023e-01 against upstream's bf16 step
for 1.049x the runtime, which is 99.6 % of the ground the host round trip buys at a tenth of the
cost. What the round trip still has over it is the forward: the renormalisation cannot fix a
softmax that was computed imprecisely, only the backward's use of it. Turning both on is safe and
pointless — a float64 softmax already sums to one, so the division is a no-op there, measured as a
bit-identical gradient.

Set `TT_BIO_SOFTMAX_BW_RENORM=0` for the old backward. **It cannot change a prediction.** Every
branch on the flag is inside a backward closure, checked by AST rather than by reading, and a
fold on OpenFold3, Protenix-v2 and OpenDDE writes the same structure with it on and off, same
card and same seed, while the counter that records the branch being reached stays at zero. It
costs 6.385e-05 s per softmax backward at 16 heads and 384 tokens, 1.0879x that op, measured
interleaved on a p150a at 1350 MHz against an A/A floor of 1.163e-05 s (`perf/of3t_d56renorm/`).

## Opt-in, and inert when off

Importing `tt_bio` does not reach the tape, the optimizer or the loss set, and
`tests/test_training_opt_in.py` enforces it. Importing `tt_bio.train` costs nothing either:
every public name resolves lazily, so `plan()`, `batches()`, `Mesh`, `AdamW` and its refusals
all work with no wheel and no card. That is Tier 0's cut line holding one level down: a dry run
answers on a laptop. The `finetune` verb is registered on the CLI group by dotted path and
loaded only when named, so `tt-bio predict` and `tt-bio --help` never pay for any of it.
