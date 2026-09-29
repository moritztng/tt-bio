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
from tt_bio import bindcraft2

with bindcraft2.campaign_predictor(card=0):
    bindcraft2.run_campaign(settings, project_folder,
                            af2_weights=params_dir, mpnn_weights=mpnn_dir)
```

`bindcraft2.run_campaign` goes where BindCraft 2's own `campaign.run_campaign` went and takes the
same arguments. It runs several design trajectories over the one chip by default, as many as your
box has the memory for, and prints the count it chose with the reason. That is worth 1.20x on a
completed trajectory and 1.22x on a round. [Several trajectories on one
card](#several-trajectories-on-one-card) below is how to read that line and how to turn it off.

`card` has to be set before ttnn is imported, because that is when ttnn reads the pin. Leave it
out to accept whatever `TT_VISIBLE_DEVICES` already says; pass it and `bindcraft2` raises rather
than silently running on the wrong chip.

## Several trajectories on one card

A design round is a host column and a device column laid end to end, and one trajectory cannot
overlap them: the card idles about 2.6 s of every round with a host thread busy in all of it.
Independent trajectories are the only work there is to fill that with, and a campaign has a
supply of them. So `bindcraft2.run_campaign` runs several of them by default.

Before any thread starts it reads free host memory, and the card when one is already open, then
takes the largest count that fits up to three. It says which it took:

```
[tt_bio.bindcraft2] 3 design trajectories on this card: 226.3 GB of host memory is free and 3 of
them peak near 20 GB. Pass trajectories_per_card to choose yourself; 1 is BindCraft 2's own loop.
```

A box whose free memory cannot be read gets one, never three. An explicit
`trajectories_per_card=N` is used exactly as given, including a number the box cannot hold, which
raises `MemoryError` naming what it wanted and what was free rather than letting the kernel kill
the campaign at round 200.

```python
# BindCraft 2's own loop: no threads, no scheduling, nothing in tt-bio behaves differently.
bindcraft2.run_campaign(settings, project_folder, trajectories_per_card=1,
                        af2_weights=params_dir, mpnn_weights=mpnn_dir)
```

What the default is worth, from two real `examples/pdl1.json` campaigns run to their stop
condition on one Blackhole chip: same commit, same seed, same six-trajectory budget, a 146-residue
binder against the 115-residue PD-L1 target at 288 tokens.

| trajectories | s per round | s per completed trajectory | accepted | chip-seconds per accepted design | host peak |
|---|---|---|---|---|---|
| 1 | 7.41 | 927.0 | 2 of 6 | 2,781 | 13.1 GB |
| 3, the default here | 6.00 | 773.8 | 2 of 6 | 2,321 | 21.9 GB |

The two campaigns accepted the same two designs, with bit-identical coordinates and the same
metrics on all six trajectories. Interleaving buys time and changes nothing else about the
result. An H200 runs this round in 0.696 s, so the default went from 10.6x that to 8.6x.

The round on its own, measured tighter: eight arms alternating in one sitting on one card, nine
rounds each, first round dropped. One trajectory reads **7.204 s** a round (7.183-7.258 across
four arms) against **5.899 s** at three (5.891-5.912), **1.221x**. AICLK was 1350 MHz in every
arm, sampled during the rounds, with no sample under 1200.

Three is the cap because a fourth bought nothing: 6.976 s a round against three at 6.992 in the
same sitting, with only 0.26 s of idle a round left to fill. How busy your host is moves the round
as much as any of this: the same three-trajectory round read 6.30 s on a box at load1 11 where a
quiet box reads 5.90. Measure your own box before comparing against anyone's number, including
these.

It costs host memory. The same campaign peaked at 13.1 GB with one trajectory and 21.9 GB with
three, about 4.4 GB for each one after the first. The resolver prices it higher than that, 8.5 GB
for the first and 6 GB for each one after, so it errs towards a campaign that is slower than it
could have been rather than one the kernel kills.

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

## What your target file can look like

BindCraft 2 reads the target structure, and tt-bio checks the inputs before it opens a card.
These all work, with the target's own residue numbering kept end to end, so a hotspot is the
residue the file calls by that number:

- **PDB and mmCIF**, and a file whose header is not the wwPDB's own.
- **Ligands, cofactors, metals, glycans and waters** left in the file. They are dropped, the
  protein chains are unchanged, and no residue number shifts.
- **Selenomethionine and the other modified residues** BindCraft 2 maps to a parent amino acid,
  and **alternate side-chain conformations**.
- **Unresolved loops.** A missing stretch stays missing; the residues around it keep their
  numbers. A hotspot inside the gap is refused rather than dropped, see below.
- **Multi-chain targets**, with `"chains": "A,B"` to design against several and `"chains": "B"`
  to pick one out of a complex. Hotspots on the second chain are written in that chain's own
  numbering, `"B125"`.
- **A FASTA target**, for a disordered one BindCraft 2 crops itself.
- **A structure pasted in as text** rather than named as a path.

A gzipped target, which is what the RCSB hands you by default, is refused with the command to
unpack it: BindCraft 2 reads a target as text. There is no fetch-by-ID; the target is a file you
have.

Insertion codes (`100A`, the antibody numbering habit) are refused: renumber first.

## Inputs that are refused

A hotspot that names no residue of the target sets no flag, and BindCraft 2's interface loss
reads "no hotspots" as "design against the whole surface". The campaign would then run to the
end and hand back confident designs against an epitope nobody chose. So `tt_bio` refuses it,
before the card is opened, naming the residue and the reason:

```
target 'T' hotspot 66: chain A of gap.pdb has no residue 66, it falls in the unresolved stretch
60-70. A hotspot is read in the file's own residue numbering. Name a residue the structure
resolves, or model the missing one in first.
```

Four spellings of the same mistake are refused this way: a residue in an unresolved loop, a
number outside the chain's range, a number written 1-based against a file that starts at 18, and
a hotspot on a chain the campaign does not design against. Coldspots are checked the same way. A
*range* that falls partly in a gap is allowed and says what it dropped, because a region is a
reasonable thing to ask for:

```
[tt_bio.bcinputs] target 'T' hotspot 54-70: 11 of 17 residues are not in the structure (60-70
unresolved); the rest carry the hotspot.
```

Two targets under one `name` are refused too, and this one is worth spelling out: BindCraft 2
keys a campaign's targets by name, so a repeated name is not a second target -- it replaces the
first. The campaign designs against the last one alone and the other target's hotspots are gone
without a word:

```
2 targets are named 'T' (il2rb.pdb, hPDL1.pdb). BindCraft 2 keys a campaign's targets by name,
so only the last of them is prepared and the rest are dropped without a word. Give each target
its own "name".
```

A target file that holds no polymer -- a ligand-only download, or a structure whose residues are
all `HETATM` -- is refused by name rather than as an `IndexError` from inside the campaign:
BindCraft 2 reads the polymer and drops every heteroatom, so such a file has no chain to design
against at all.

Two settings are refused for the same reason, that they would otherwise cost a whole campaign:
a confidence threshold written as a percentage (`"min_plddt_final": 80`, where the scale is 0 to
1) accepts nothing however long it runs, and `"binder_lengths": 80` is not a list, which stops
BindCraft 2 inside its length sampler. Write `[80]`, or `[60, 90]` for a range.

An NMR ensemble is not refused: BindCraft 2 reads `MODEL 1` and ignores the rest, so a
20-model ensemble designs against its first model. Split the model you want out first if that is
not the one you meant.

Numbering is the file's own throughout, never a 1-based position. A target renumbered by a
modelling tool is self-consistent and cannot be told apart from the original, so hotspots move
with it: check the numbering of the file you hand in, not the one you downloaded.

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
runs. They were measured before the gradient kernels and the interleaving default above, so treat
that as an upper bound rather than as today's cost: on the current tree the six-trajectory PD-L1
campaign in the table above accepted two designs at **2,321 chip-seconds each**. Six trajectories is far too few to put an
acceptance rate on; read that number as a cost per accepted design on this tree, not as evidence
that the loop accepts more often.

One caution when you read your own verdicts, and it is BindCraft 2's behaviour rather than the
card's: `predicted_tm_score` is a maximum over the PAE rows, so a single collapsed row pins pTM
and i_pTM at that length's ceiling on either arm. Grade a `mutate`-stage verdict on pLDDT.

## Licence

BindCraft 2 ships under a source-available, hosting-restricted licence and tt-bio neither vendors
it nor exposes it through any hosted service. There is no JapanFold route to BindCraft 2 and no
catalog entry for it. Read BindCraft 2's licence and decide for yourself whether your use is
inside it.
