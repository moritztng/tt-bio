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
amortises over hundreds of steps. A steady-state per-step cost has not been measured on this tree,
so this page does not quote one. Plan a campaign on the assumption that the first trajectory is
much slower than the ones after it, and measure your own steady state before sizing a run.
[`docs/gradient-step-cost.md`](gradient-step-cost.md) prices one gradient step through the
same taped pair track at AlphaFold 2's dimensions, which is the closest thing to a per-step
budget anyone has measured here.

## What is not settled

The gradient loop runs on card and the design loop completes trajectories. **Design acceptance is
still being qualified.** The device arm has zero valid acceptance readings against BindCraft 2's
shipped five-model pool: the two completed device trajectories both ran a checkpoint mismatch that
has since been fixed, and the first clean trajectory after the fix was rejected at the `mutate`
stage on a defect still under investigation. BindCraft 2's own JAX reference accepted 1 design in
1 completed trajectory on the same settings.

So this page says the loop runs, at these sizes and this cost. It does not say the designs are
good, and you should qualify that yourself before trusting a run. Both counts above come from the
default path, with validation on BindCraft 2's own JAX trunk.

One rough edge: closing the card at the end of a process that has also run JAX can abort in the
driver, with `pthread_mutex_unlock failed for mutex CHIP_IN_USE_0_PCIe`. It happens after the work
is finished, the chip is left healthy, and the results already written are valid, but the process
exit status is a crash. Write your outputs out as you go rather than at exit.

## Licence

BindCraft 2 ships under a source-available, hosting-restricted licence and tt-bio neither vendors
it nor exposes it through any hosted service. There is no JapanFold route to BindCraft 2 and no
catalog entry for it. Read BindCraft 2's licence and decide for yourself whether your use is
inside it.
