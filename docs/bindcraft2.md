# Running BindCraft 2 on a Tenstorrent card

[BindCraft 2](https://github.com/PacesaLab/BindCraft2) designs binders by differentiating
AlphaFold 2 through a sequence. `tt_bio.bindcraft2` gives it an Evoformer trunk that runs on a
Blackhole chip. BindCraft 2 keeps everything else: its padding, templates, input features,
structure module, confidence heads, Kabsch alignment, filters, ranking and step budget. Only
AlphaFold 2's 48 Evoformer blocks move, which is where every O(L^3) op in the trunk lives.

tt-bio does not ship BindCraft 2 and does not serve it. This page is for an installation you run
yourself, and BindCraft 2's own licence governs what you may do with it.

## What you need

- BindCraft 2 installed and importable.
- AlphaFold 2 parameters: `tt-bio weights --download af2ig` puts `params_model_1_ptm.npz` in
  tt-bio's weights cache, which is also a valid `data_dir` for BindCraft 2. A multi-model campaign
  needs one `params_<model>.npz` per model it draws from, in one directory.
- One Blackhole card, pinned. ttnn brings up every card `TT_VISIBLE_DEVICES` names, not just the
  one it computes on, so pin it to the chip you were given.

## Run a campaign

`campaign.py` constructs its predictor in one place, so rebinding that one name is the whole
integration:

```python
import bindcraft.campaign as campaign
from tt_bio import bindcraft2

with bindcraft2.campaign_predictor(card=0):
    campaign.run_campaign(settings, project_folder,
                          af2_weights=params_dir, mpnn_weights=mpnn_dir)
```

`card` has to be set before ttnn is imported, because that is when ttnn reads the pin. Leave it
out to accept whatever `TT_VISIBLE_DEVICES` already says; pass it and `bindcraft2` raises rather
than silently running on the wrong chip.

## Several trajectories on one card

A design round is a host column and a device column laid end to end, and one trajectory cannot
overlap them: the card idles about 2.6 s of every round with a host thread busy in all of it.
Independent trajectories are the only work there is to fill that with, and a campaign has a
supply of them.

```python
from tt_bio import bindcraft2

with bindcraft2.campaign_predictor(card=0, exact=False, extra_msa=True, template=True):
    bindcraft2.run_campaign(settings, project_folder, trajectories_per_card=3,
                            af2_weights=params_dir, mpnn_weights=mpnn_dir)
```

This is the call the round below was measured with. `exact=False`, the default, also turns on
tt-bio's gradient kernels for BindCraft 2 (`fast=True` is its default there), and puts them back when the
block exits, so nothing else in the process runs differently. No environment variable is
involved.

`bindcraft2.run_campaign` goes where `campaign.run_campaign` went. At the default
`trajectories_per_card=1` it *is* that call: no threads, no scheduling, nothing in tt-bio behaves
differently. Above 1 it runs that many trajectories on their own threads over one chip, each
claiming its own trajectory number out of the project's progress file exactly as separate worker
processes would.

On one Blackhole chip, a 288-token PD-L1 round:

| trajectories | s per round, amortised over the trajectories running |
|---|---|
| 1 | 9.06 |
| 2 | 7.23 |
| 3 | 6.76 |

An H200 runs the same round in 0.696 s. Every figure here was taken at a 1350 MHz AICLK, sampled
during the rounds. One more gradient kernel landed after the table: with it, three trajectories
run at 6.18 s and two at 6.62 s, the latter measured with the predictor arguments above and no
environment variable set. What the option is worth to you depends on how much of your round is
host time, since that is all it fills.

How busy the host is moves the round as much as any of this. On a quiet box, load1 2.4-4.0, the
three-trajectory round reads **5.93 s**; the tree it was measured against read 5.86 s in that same
sitting and 6.30 s in an earlier one at load1 11. Same p300c box, 1350 MHz in all of them, so the
7 % is host load and nothing else. Measure your own box before comparing against anyone's number,
including these.

The rate holds over a whole campaign, not just a burst: four PD-L1 trajectories at
`trajectories_per_card=2`, run to their stop condition, took 500 gradient rounds at 6.78 s
amortised, 2.2 % under a 9-round measurement on the same chip.

It costs host memory: about 3.5 GB per trajectory beyond the first, on top of the roughly 8 GB one
trajectory of this size holds. Two of them peaked at 14.2 GB, three at 19.5 GB. Both the box and
the card are read before any thread starts, and a box that cannot hold them raises `MemoryError`
naming what it wanted and what was free, rather than letting the kernel kill the campaign at
round 200.

Trajectory *i* starts only once *i-1* has its first gradient round behind it, so no two of them
compile at the same time. Their output interleaves on stdout.

## Or build one predictor

```python
with bindcraft2.predictor(card=0) as build:
    model = build(presets="model_1_ptm", data_dir=params_dir, length_bucket_size=32)
    predictions, gradients, loss = model.sequence_gradients(protein_states, losses)
```

`build` takes BindCraft 2's own `AlphaFoldDesignModel` arguments and returns a
`DifferentiableProteinPredictor`.

## Several checkpoints

BindCraft 2 samples one design model per gradient step, so a campaign with a five-model pool wants
all five on card. Pass the directory and the pool fills itself from the models BindCraft 2 asks
for:

```python
with bindcraft2.campaign_predictor(card=0, checkpoints="/path/to/af2/params", resident=1):
    ...
```

`resident` caps how many trunks stay on card and evicts least-recently-used. Five AF2 trunks is
about 910 MB of weights. Holding all five brought a backward-pass allocator refusal forward at 288
tokens that `resident=1` ran past, so set it if a long run dies in the allocator; the reload it
costs is smaller than a step.

A checkpoint the directory does not have folds on BindCraft 2's own trunk instead of stopping the
campaign, and the first such fold says so on stdout. You will see this even with a complete
directory, because the validation ensemble is deliberately kept off the card (below).

The choice is made per model family rather than per checkpoint, and that is not a detail you can
ignore when sizing a weights directory: BindCraft 2 compiles one program per family, so the five
multimer checkpoints are all-or-nothing and so are the two monomer ones. Give it three of the five
multimer files and all five design folds move to the host. Either hold a family completely or
expect it on the host.

If nothing you asked for is on card, building the predictor raises rather than falling back. A
campaign that runs entirely on the host while you believe it is on a card produces numbers you
would then misattribute.

## Which folds run on the card

The design loop runs on card. **The validation ensemble runs on BindCraft 2's own JAX trunk by
default**, and that is the setting any accepted count should be quoted from.

Validation is what decides whether a design is accepted. Folding it on card would put device
numerics inside the instrument that grades the device; on the host trunk that stage is bit-for-bit
BindCraft 2's own, so the thing being measured stays the gradient loop. Validation is a few
forward folds per completed trajectory against the trajectory's own gradient steps, so it is not
where a campaign spends its time; what that costs on your host has not been measured here.

```python
with bindcraft2.campaign_predictor(card=0, validation="device"):  # both on card
    ...
```

`validation="device"` is a reasonable choice for throughput, but an accepted count measured that
way is a different measurement. Re-measure before quoting it, and say which path produced it.

### The extra-MSA stack

BindCraft 2 runs a four-block extra-MSA stack before the Evoformer. It stays in JAX unless you ask
for it:

```python
with bindcraft2.predictor(card=0, extra_msa=True) as build:
    ...
```

Off by default, and switchable independently of the Evoformer, so a comparison graded on the
Evoformer alone keeps the program it was graded on.

Before tt-bio's gradient kernels it made a round slower: the card ran the stack in 34.0 s where
BindCraft 2's JAX ran it on the host in 10.3 s, about 4 % on the round, measured with both arms
interleaved in one process on one card, 16 rounds, seven per arm. The interleaved round in the
table above was measured with it on and has not been re-measured with it off.

The reason is what the Evoformer swap already did. With the Evoformer on the card a round is 97 %
device time and only 13 s of 454 s is left on the host, so even a free extra-MSA swap could win
2 %. The two levers do not add up: the first one takes the host time the second one was going to
save. Read `build.extra_msa.calls` to confirm the card ran it.

BindCraft 2 feeds an all-zero extra-MSA mask, so there is no gradient into the extra MSA to lose:
it measures exactly zero on BindCraft 2's own JAX, and the card's path returns zero by
construction and refuses a mask that is not all-zero.

### The exact-training instrument

Inside a tape, tt-bio can run softmax and layer norm on the host in float64 rather than on the
device. That is what reproduces AlphaFold 2's own gradient most closely, and a BindCraft 2 round
pays a host round trip for every one of them. It is off by default; ask for it with:

```python
with bindcraft2.predictor(card=0, exact=True) as build:
    ...
```

At n=192 a round makes 576 exact softmax calls, 1,728 exact layer-norm calls and 576 layer-norm
backward VJPs, and pushes 7.46e9 layer-norm elements through float64. One `sequence_gradients`
call costs 479.59 s with the instrument on against 19.285 s with it off, 24.87x, measured
interleaved in one process on one card at AICLK 1350 sampled during the call. That is the
gradient call and not the whole design round, which also carries BindCraft 2's own JAX work.

What it buys is small here. Against a float64 reference, the worst gradient tensor sits at
rel_l2 0.087998 with the instrument on and 0.088985 with it off, a move of 1.1 %, where
bfloat16 alone already carries 0.075483 of that distance. A design campaign with the instrument
off accepted a 93-residue binder on the shipped `examples/pdl1.json` clearing all seven of
BindCraft 2's final filters, at pLDDT 0.90, i_pTM 0.79, zero backbone clashes and hotspot
contact fraction 1.0.

So it is off by default: 460 s a round buys 1.1 % of an error budget bfloat16 already owns 85 %
of, and a design loop is graded on the binders it accepts rather than on gradient distance.
Set `exact=True` to reproduce a training-style gradient bar, which BindCraft 2 does not have.
Read `tt_bio.autograd.EXACT_SOFTMAX_STATS` and `EXACT_LAYER_NORM_STATS` to confirm which one
ran: on the default both stay at zero.

Turning it off changes only how softmax and layer norm are computed inside the tape. It does not
skip a step, a recycle or a block.

The default, `exact=False`, also arms tt-bio's gradient kernels for the duration of the predictor,
the ones the round figures on this page were measured with. Pass `fast=False` to keep the device softmax and
layer norm without them. Like `exact=False`, they change which kernels compute the round and do
not skip any of its work.

## The control arm

`trunk="jax"` opens no device and touches no card. It runs BindCraft 2's own trunk through the
same class, the same call path and the same bucket rounding, which is what a device result should
be read against rather than a differently shaped program.

```python
with bindcraft2.predictor(trunk="jax") as build:
    ...
```

## What fits

Every binder length BindCraft 2 draws against the 115-residue PD-L1 target fits on one card. The
draw range is 60 to 180 residues, three roundings collapse it to five padded sizes (192, 224, 256,
288, 320 tokens), and the largest runs six consecutive gradient steps with the resident DRAM line
flat. Peak allocation at the top draw is 19.19 GB of the card's 34.226 GB, in the backward, with
12.68 GB still free. Measured on a p300c at AICLK 1350 median over 808 samples taken during the
step.

The token axis buckets to 32 and rounding up is faster, not slower: the PD-L1 complex at 211
tokens costs 4.504 s on the trunk forward and the same design padded to 224 costs 1.369 s,
same card and same AICLK 1350 median.

## What one step costs

One gradient step through this entry point on a real card, at the small end of the draw range:

| | |
|---|---|
| complex | 115-residue PD-L1 target + 60-residue binder = 175 tokens, padded to 192 |
| trunk calls | 2 device forwards under the tape, 1 device backward, 0 tapes left on card |
| first call, end to end | 1304.6 s |
| card | p150a, AICLK median 1350 MHz over 354 samples polled during the step |
| host | 4 cores of a box at load/core 1.39 |

**That 1304.6 s is a first call, not a step time.** It carries the AlphaFold 2 weights onto the
card and JAX's compile of the whole design program, both of which a campaign pays once and then
amortises over hundreds of steps. The steady-state cost is the round table above, which is two
thirds of the way up the draw range rather than at the bottom of it. Plan a campaign on the
assumption that the first trajectory is much slower than the ones after it.
[`docs/gradient-step-cost.md`](gradient-step-cost.md) prices one gradient step through the
same taped pair track at AlphaFold 2's dimensions, which is the closest thing to a per-step
budget anyone has measured here.

## Do the designs pass?

On the shipped `examples/pdl1.json`, the card accepts binders at a rate BindCraft 2's own JAX does
not separate from:

| arm | accepted / completed trajectories | rate | 95 % CI |
|---|---|---|---|
| this trunk, on card | 7 / 31 | 0.226 | 0.096 - 0.411 |
| BindCraft 2's own JAX | 1 / 5 | 0.200 | 0.005 - 0.716 |

Fisher exact, two-sided: **p = 1.00**. Both arms ran the same settings file and the same filters,
and a design is accepted only by BindCraft 2's own final filters, never by anything tt-bio wrote.
24 of the 31 device trajectories ran the default path with validation on BindCraft 2's JAX trunk
and accepted 5; the other 7 folded validation on the device pool and accepted 2. Those two do not
separate either (p = 0.64).

**Read the interval, not the point.** Thirty-one trajectories against five is enough to say that
nothing visible is broken and not enough to certify a small difference: telling 0.226 from 0.200
apart, if the gap were real, needs roughly 109 trajectories per arm. The reference arms kept
running past the five this comparison was committed to and accepted 4 of the next 7. That block is
not part of the pre-registered comparison and does not separate from the card either (p = 0.16),
but it runs above our rate rather than below it, so it is reported here rather than dropped.

Cost, on those same 31 trajectories: **22,599 chip-seconds per accepted design**, 5,103 per
completed trajectory, across p150a and p300c cards at an AICLK of 1350 MHz sampled during the
runs. They were measured before the gradient kernels the round table above was measured with, so
treat that as an upper bound on what a design costs today.

One caution when you read your own verdicts, and it is BindCraft 2's behaviour rather than the
card's: `predicted_tm_score` is a maximum over the PAE rows, so a single collapsed row pins pTM
and i_pTM at that length's ceiling on either arm. Grade a `mutate`-stage verdict on pLDDT.

One rough edge: closing the card at the end of a process that has also run JAX can abort in the
driver, with `pthread_mutex_unlock failed for mutex CHIP_IN_USE_0_PCIe`. It happens after the work
is finished, the chip is left healthy, and the results already written are valid, but the process
exit status is a crash. Write your outputs out as you go rather than at exit.

## Licence

BindCraft 2 ships under a source-available, hosting-restricted licence and tt-bio neither vendors
it nor exposes it through any hosted service. There is no JapanFold route to BindCraft 2 and no
catalog entry for it. Read BindCraft 2's licence and decide for yourself whether your use is
inside it.
