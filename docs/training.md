# Training: the four tiers, and what each one guarantees

Progressive disclosure, cut where the **unit of user authorship** changes. Not where the
amount of configuration changes, which is the distinction this whole design turns on: you can
name your authorship unit before you start, while "how much configuration" is only knowable
after you hit a wall.

| Tier | Surface | You own | Cut line, and its test |
|---|---|---|---|
| 0 | `tt-bio finetune ...` | a config | no callables in the signature |
| 1 | `train.finetune(...) -> Run` | the objective | no `for` over steps in your code |
| 2 | `plan`, `batches`, `objectives`, `AdamW`, `Checkpointer`, `Mesh`, `LoraConfig`, `lora_factors_for`, `attach` | the `for` statement | no `ttnn` call in your code |
| 3 | `tt_bio.autograd` + `train.gradcheck` | an op and its backward | `ttnn` appears here |

Each cut line has a test in `tests/test_train_interface.py` that can fail. A boundary that is
only described drifts, and three of the four tests below caught a real defect while being
written: the Tier-0 dry run was importing the tape, `from tt_bio.train import plan` was handing
back a module where the tier promised a callable, and the naming collision was flagged by a
gate rather than by taste.

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

**The diffusion module's fp32 is the forward's boundary, not the backward's.** Measured on one
Protenix token-DiT block against a float64 reference, `perf/ptx_diffusion/`: bf16 activations
cost 4.25e-02 worst relative L2 across 22 gradients against fp32's 1.65e-02, at cos 0.9991, and
a 150-step fit of a fixed target still drops the loss 2076x where fp32 drops it 2869x. bf16
activations train. Neither the accumulator nor the attention site is the lever -- dropping the
backward config from `precise()` costs 9 %, and raising only the attention matmuls and the
softmax back buys 5 % -- so the error is operand rounding and no kernel setting reaches it. The
reason `PROTENIX_DIFFUSION_FP32_DEVICE` stays on for training is that the forward we serve is
fp32, and a training forward that is not the served forward is the second implementation
`tt_bio/ops.py` exists to prevent. `PROTENIX_DIFFUSION_FP32_DEVICE=0` is a measured lever with
a quoted cost, not an unknown.

**A bf16 master stalls, and the stall is in the master, not the device copy.** Same block, same
seed, 150 Adam steps at lr 1e-3: with an fp32 master the fraction of master elements a step
moves stays 1.000 throughout; with a bf16 master it falls 1.000 -> 0.361 by step 149 while the
loss curve stays the prettiest of the three arms. On the same block's real gradient at Adam's
step-1 magnitude over 8,261,376 elements, a bf16 master keeps 0.9999 of its updates at lr
1.8e-3, 0.605 at lr 1e-4 and 0.086 at lr 1e-5, and an fp32 master keeps 1.000 at all of them.
Since Adam's effective step shrinks as `m/sqrt(v)` falls toward convergence, every run reaches
the stalling regime from above whatever lr it started at. This is the master ratio, not the
device copy's -- the per-step scatter the paragraph above describes is a property of the copy.

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
- The **diffusion module's backward costs 3.036x its forward**, 44.0 ms against 133.6 ms for the
  24-block token DiT at a 384-token crop in fp32, 1350 MHz sampled during, A/A floor 0.22 %,
  reproduced across two runs. Protenix differentiates one denoise call per step, so that is
  89.6 ms added per training step.
- The same backward **retains 4.25 GiB**, 4.324 GiB peak against 0.077 GiB forward-only, i.e.
  190 MB per block and 12.6 % of the card. bf16 activations halve it.

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
its chips share a host or not. Until it is wired, the honest claim is multi-card on one host.

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

## Adapting a model: how the sites are found

`lora_factors_for` runs the shipped forward once with a census hook installed. Every call that
routes through `tt_bio.ops.linear` announces itself with its own shapes; a call that does not
route through it is not adaptable, and the census says so by not listing it. The hook records
and then declines every call, so a census pass computes exactly what an inference pass computes
and the output is comparable byte for byte.

Sites are named by call site (`file:line:qualname`) rather than by weight identity, because the
same weight is read at one site while two different weights are read at one shared helper. The
call site is the thing you can find and target with `--target`.

`attach` composes over whatever hook is already installed instead of replacing it. A frozen
trunk still has to stay taped downstream of its first adapter or the gradient never reaches the
adapters in the early layers, so declining a non-adapter site outright would train only
whatever sits after the last adapter, with no error and a loss curve that still falls.

## Opt-in, and inert when off

Importing `tt_bio` does not reach the tape, the optimizer or the loss set, and
`tests/test_training_opt_in.py` enforces it. Importing `tt_bio.train` costs nothing either:
every public name resolves lazily, so `plan()`, `batches()`, `Mesh`, `AdamW` and its refusals
all work with no wheel and no card. That is Tier 0's cut line holding one level down: a dry run
answers on a laptop. The `finetune` verb is registered on the CLI group by dotted path and
loaded only when named, so `tt-bio predict` and `tt-bio --help` never pay for any of it.
