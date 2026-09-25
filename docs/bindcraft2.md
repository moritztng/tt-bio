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

BindCraft 2 samples one design model per gradient step, so a campaign with a five-model pool needs
all five on card. Pass the directory and the pool fills itself from the models BindCraft 2 asks
for; a model it draws that the directory does not have is an error, not a fallback:

```python
with bindcraft2.campaign_predictor(card=0, checkpoints="/path/to/af2/params", resident=1):
    ...
```

`resident` caps how many trunks stay on card and evicts least-recently-used. Five AF2 trunks is
about 910 MB of weights. Holding all five brought a backward-pass allocator refusal forward at 288
tokens that `resident=1` ran past, so set it if a long run dies in the allocator; the reload it
costs is smaller than a step.

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
tokens costs 4.504 s on the trunk forward and the same design padded to 224 costs 1.369 s.

## What is not settled

The gradient loop runs on card and the design loop completes trajectories. **Design acceptance is
still being qualified.** The device arm has zero valid acceptance readings against BindCraft 2's
shipped five-model pool: the two completed device trajectories both ran a checkpoint mismatch that
has since been fixed, and the first clean trajectory after the fix was rejected at the `mutate`
stage on a defect still under investigation. BindCraft 2's own JAX reference accepted 1 design in
1 completed trajectory on the same settings.

So this page says the loop runs at these sizes and these speeds. It does not say the designs are
good, and you should qualify that yourself before trusting a run.

## Licence

BindCraft 2 ships under a source-available, hosting-restricted licence and tt-bio neither vendors
it nor exposes it through any hosted service. There is no JapanFold route to BindCraft 2 and no
catalog entry for it. Read BindCraft 2's licence and decide for yourself whether your use is
inside it.
