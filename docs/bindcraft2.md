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

Before any thread starts it prices a trajectory at your design's token axis, reads free host
memory and the card, and takes the largest count that fits up to three. It says which it took:

```
[tt_bio.bindcraft2] 3 design trajectories on this card: 3 of them at 288 tokens hold about 16 GB
of the card and peak near 20 GB of the 226.3 GB of host memory free. Pass trajectories_per_card
to choose yourself; 1 is BindCraft 2's own loop.
```

**The count falls as the design grows.** A trajectory's memory grows with the square of the
token axis, and at some axes the fused triangle attention does not fit the chip and a slower path
holds about twice as much: at 544 tokens one trajectory holds 21.5 GB of a p300 chip where 512
holds 10.4. Every figure in this section is that chip, and the board moves them a little: the
p150a in "What fits" below holds 25.75 GB at the same 544-token axis.
Which axes do that is only known once the card has tried, and a campaign draws several
binder lengths, so the default prices every design as if it were one of them. On a 32 GB
Blackhole chip that is three trajectories up to 352 tokens, two at 384 and 416, and one from 448
up. Where one fits, the default is one, which is BindCraft 2's own loop unchanged.

**The part matters, and the default reads it.** A Wormhole chip has 12 GB where a Blackhole chip
has 32, so the counts above are not the counts there: at 288 tokens the default takes one, or two
when a chip is already open and can report its own free memory. Two at 288 tokens has run a
campaign to its stop condition on a Wormhole Galaxy chip, so pass `trajectories_per_card=2` if
your designs stay near that size. Three is refused on that part.

So the size the line prints is an upper bound, and above about 448 tokens it can be roughly
twice what the run goes on to hold: a 576-token design is priced at 29.9 GB and holds 13.1 GB
when the fused arm serves. It is not telling you the card is too small. It is saying it will
not start a second trajectory on a design that might need the slower path.

A box whose free memory cannot be read gets one, never three, and so does a design whose token
axis cannot be read. An explicit `trajectories_per_card=N` is used exactly as given, including a
number that will not fit, which raises `MemoryError` naming what it wanted and what was free
rather than dying in the middle of a round.

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

### The extra-MSA stack and the template embedder

BindCraft 2 runs a four-block extra-MSA stack before the Evoformer, and a multimer template
embedder with two more pair blocks. Both run on the card by default. Pass False to leave either
in BindCraft 2's JAX:

```python
with bindcraft2.predictor(card=0, extra_msa=False) as build:
    ...
```

Leaving them in JAX costs far more on the host than running them costs on the card. On a Wormhole
Galaxy chip a 288-token round is 29.423 s with both in JAX against 16.267 s with both on card,
1.8087x: the host column falls from 17.188 s to 2.417 s and the device column rises from 12.294 to
13.856. Eight arms alternated at the process boundary in one sitting on one chip, AICLK 1000 MHz
median with none of the 454 samples below it. On a Blackhole p150a chip the same change is
25.762 against 8.495 s, 3.03x, at AICLK 1350.

That is one trajectory. Two interleaved trajectories hide most of the host column behind each
other's device time, so on a Wormhole chip at its default of two the same change is 15.776 against
13.923 s a round, 1.1331x. A Blackhole chip runs three at 288 tokens, and there it is 9.564
against 6.278 s a round, 1.52x.

The designed structure moves by 0.54 A on the confident core, inside the 0.60 A bar the other
levers are held to.

Early on this went the other way: before tt-bio's gradient kernels the card ran the extra-MSA
stack in 34.0 s where JAX ran it on the host in 10.3 s. The kernels closed that gap, and the
host cost is what is left.

The template swap brings two more blocks of weights per checkpoint onto the card, so it is not
free of allocator pressure. Monomer checkpoints are untouched; the swap is installed on the
multimer modules only. Read `build.extra_msa.calls` and `build.template.calls` to confirm the
card ran them.

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

A chain named twice in one target's `chains` is refused, because `merge_receptor_chains`
concatenates the chains it is given in the order it is given them: `"chains": "A,A"` on the
115-residue hPDL1 prepares a 230-residue target and flags hotspot 54 at both 54 and 217, a
homodimer nobody asked for with the epitope on both halves of it. `A,B`, `A, B`, `B,A` and a
single chain are all fine; a target that really is a homodimer needs a file holding both copies.

The order the chains are listed in is the order they are fused, so it decides the numbers a
hotspot lands on: on the two-chain target, `A57` resolves to 57 with `"chains": "A,B"` and to
320 with `"chains": "B,A"`. Both name the same residue of chain A -- the number to read a
hotspot against is always the file's own, and the fused index is internal.

A hotspot that names no chain is noted rather than refused, on a target whose chains share that
residue number. BindCraft 2 qualifies a bare span with the target's first chain, which is the
right default and says nothing about the alternative -- and most deposited complexes number
every chain from 1, an antibody's heavy and light chains included. Chains A and B of BindCraft
2's own `hIL2R_beta_gamma.pdb` both hold residues 57-59:

```
[tt_bio.bcinputs] target 'T' hotspot 57 names no chain, and chains A, B of
hIL2R_beta_gamma.pdb each hold those residues. BindCraft 2 takes the first, chain A, and that is
the one carrying the hotspot. Write A57 to say so, or B57 for the other one.
```

`A57` resolves to 57 and `B57` to 283, through the chain-break offset the receptor fusion
applies, so the two spellings do name different residues.

Two targets under one `name` are refused too, and this one is worth spelling out: BindCraft 2
keys a campaign's targets by name, so a repeated name is not a second target -- it replaces the
first. The campaign designs against the last one alone and the other target's hotspots are gone
without a word:

```
2 targets are named 'T' (il2rb.pdb, hPDL1.pdb). BindCraft 2 keys a campaign's targets by name,
so only the last of them is prepared and the rest are dropped without a word. Give each target
its own "name".
```

A hotspot that points at a ligand, a metal, a glycan or a water is refused by name -- `residue
401 of chain A in ligands.pdb is NAG, a heteroatom` -- because those are on screen in a viewer
with residue numbers of their own and BindCraft 2 designs against the polymer only. The three
ways a hotspot can miss (a heteroatom, an unresolved stretch, a number the chain never reaches)
each say which one it is.

A **nucleotide** sequence pasted in where the protein sequence goes is refused. BindCraft 2
names every letter that is not an amino acid -- `X`, `*`, a stray digit, the `U` of RNA -- but
DNA is spelled entirely in letters that are: A, C, G and T are also alanine, cysteine, glycine
and threonine, so a coding sequence copied out of a genome browser folds as a poly-Ala/Cys/Gly/
Thr peptide of the same length, hotspots and all, and nothing downstream can tell.

```
target 'D': cds.fasta reads as a nucleotide sequence rather than a protein one. All 60 of its
letters are A, C, G or T (A 16, C 12, G 13, T 19) -- which are also the codes for alanine,
cysteine, glycine and threonine [...] Translate the sequence to amino acids first. If it really
is a protein of only those four residues, hand it in as a structure file, which is not read this
way.
```

The test is narrow on purpose: every letter one of ACGT, three of the four present, each at
least a tenth of the sequence, at least 30 residues. Poly-alanine, (GA)n elastin-like and (GT)n
repeats are real designs spelled in nucleotide letters and are not refused, nor is anything
under 30 letters. A lower-case FASTA is read exactly like an upper-case one.

A hotspot on a **FASTA** target is refused unless the crop is turned off, and this is the one
that costs the most for the least visible reason. BindCraft 2 crops a sequence target to a window
sampled at random -- `crop_fasta_sequence`, which defaults to 10-40 residues for every FASTA --
and it applies the crop by slicing the hotspot flags along with the sequence. So the hotspot a
researcher wrote is kept or thrown away by a dice roll, and when it is thrown away the campaign
is the no-epitope case again. Measured here on a 60-residue target with hotspots `5,50` over six
campaign seeds: five kept no hotspot at all, the sixth kept one of the two. The single line
BindCraft 2 prints, `target=T crop=8-35/60`, says nothing about hotspots.

```
target 'T': this is a FASTA target with '5,50' asked for, and BindCraft 2 crops a FASTA target
to a window of 10-40 residues sampled at random out of its 60. [...] Write
"crop_fasta_sequence": false to keep the whole sequence, or a crop as long as the target, or
hand in a structure.
```

Cropping a sequence target with no hotspots is the feature and is untouched, as is a crop as long
as the target -- the shape BindCraft 2's own IDR example ships.

A file whose extension does not match its records is refused with the rename to make, because
BindCraft 2 picks its reader from the suffix: a PDB saved as `.cif` fails inside the mmCIF reader
(`There are no blocks in the file`) and an mmCIF saved as `.pdb` fails inside the PDB one
(`Illegal hybrid-36 string`).

A residue numbered at or below zero -- a structure deposited with its expression tag still
numbered -3, -2, -1, 0 -- cannot be named as a hotspot at all: a span writes a range as
`A35-40`, so the minus is the separator. The refusal says so and names the file's own first
residue. Renumber the file from 1 if you need those residues.

A target file that holds no polymer -- a ligand-only download, or a structure whose residues are
all `HETATM` -- is refused by name rather than as an `IndexError` from inside the campaign:
BindCraft 2 reads the polymer and drops every heteroatom, so such a file has no chain to design
against at all.

Two settings are refused for the same reason, that they would otherwise cost a whole campaign:
a confidence threshold written as a percentage (`"min_plddt_final": 80`, where the scale is 0 to
1) accepts nothing however long it runs, and `"binder_lengths": 80` is not a list, which stops
BindCraft 2 inside its length sampler. Write `[80]`, or `[60, 90]` for a range.

That second one is raised before `tt_bio` is called at all -- a campaign is loaded first, and a
scalar length stops inside `load_settings` as `TypeError: 'int' object is not iterable`, which
names neither the setting nor the fix. So load through the check to get the message:

```python
settings = bcinputs.load_settings(request)          # BindCraft 2's own, message first
```

It is `bindcraft.settings.load_settings` in every other respect, asserted against it on a
correct campaign, and checks only what it can read off the request as written.

Two settings that BindCraft 2 accepts are refused here because of what they do far from where
they were written: a negative recycle count (`design_recycles`, `validation_recycles`,
`betasheet_reopt_recycles`) reaches JAX as an invalid tensor dimension and fails inside the first
fold with the card already open -- write `0` if you want a single pass with no recycling -- and
`"number_of_final_designs": 0` ends a campaign before it takes a trajectory, because BindCraft 2
stops as soon as the accepted count reaches it. Pass `trajectory_only` with `max_trajectories`
if trajectories without acceptance is what you wanted.

An NMR ensemble is not refused: BindCraft 2 reads `MODEL 1` and ignores the rest, so a
20-model ensemble designs against its first model. Split the model you want out first if that is
not the one you meant.

An mmCIF's two numberings do not have to agree, and BindCraft 2 reads the **author** one:
`auth_seq_id` and `auth_asym_id`, not `label_seq_id` (which the wwPDB numbers from 1) or
`label_asym_id`. So a hotspot means the same residue in the mmCIF and the PDB of the same entry,
checked both ways round.

Numbering is the file's own throughout, never a 1-based position. A target renumbered by a
modelling tool is self-consistent and cannot be told apart from the original, so hotspots move
with it: check the numbering of the file you hand in, not the one you downloaded.

## What fits

Two things decide whether a campaign runs: whether the input is one BindCraft 2 can read, which
is the two sections above, and how big the complex is, which is this one.

Size is counted in **tokens**: the residues of the fused complex BindCraft 2 builds, rounded up
to a multiple of 32. That is not your target length plus your binder length. The fusion adds 18
to 28 residues, so a job sized off residue counts comes out one or two buckets low. A 387-residue
target with a 146-residue binder is a 576-token job, not a 544-token one.

Measured end to end on a p150a across eight deposited targets, 115 to 674 residues, one to three
chains and six folds, with the card's AICLK sampled during each run and its median 1350 MHz in
all 445 of them:

| tokens | outcome | peak DRAM of 34.226 GB | one gradient round |
|---|---|---|---|
| 192 to 512 | runs and completes | 2.63 to 11.42 GB | 5.89 to 34.19 s |
| **544** | runs and completes, at 1.8x the memory and 2.0x the round of either neighbour | 25.75 GB | 67.27 s |
| 576 | runs and completes | 14.23 GB | 47.38 s |
| 608 and above | refused, naming the axis and the memory | | |

**The card is never the limit inside the supported range.** At the top of it the job holds 42 %
of the board. What stops 608 is a single `[608, 4, 608, 608]` fp32 score tensor on the composed
path, 3.6 GB asked for against 4.1 GB free but only 327 MB contiguous, after the fused
triangle-attention arm declined every call. At 736 the card genuinely is full, 243 MB free
against a 277 MB request, and the message says which of the two it is rather than calling
everything fragmentation. Both refusals raise; nothing in the range OOM-killed a process, hung
one, or returned a wrong answer.

**544 tokens is the axis to avoid**, and the reason is the fused triangle-attention forward. It
runs at 288 through 512 and again at 576, and at 544 it does not fit the chip, so the round falls
back to the composed path: 25.75 GB and 67.27 s against 11.42 GB and 34.19 s at 512 and 14.23 GB
and 47.38 s at 576. Both the memory and the round time jump together, which is the signature. The
run says so itself when it happens and says which way to move; moving the axis by one bucket in
either direction is the fix.

The *backward* is a second kernel with its own, narrower limit, and it is the expensive half of a
gradient round. Counted on the p150a ladder at 108 backward calls a round, it serves all 108 at
288 and **0 of 108** at 352, 416, 448, 480, 512 and 576: above 288 the gradient runs on the
chunked recompute at every size measured. The 576 campaign below confirms it at length, 0 of
27,000 calls served over 249 rounds. So "the fused arm runs at this size" is a statement about the
forward, and a round four times the length of the 288-token one is not a round that failed.

The arm also serves nothing at 192, where it costs nothing worth noticing: that is the cheapest
rung on the ladder at 2.63 GB and 5.89 s. Declining is a fallback, not a failure. Nor is it
all-or-nothing at a given size: over the whole 576-token campaign the fused forward **served
81,000 calls and declined 15,232**, one call in six, and the campaign carried those through the
composed path and kept designing. The 288-token campaign beside it declined nothing at any stage.
So above about 512 tokens expect a share of the calls on the composed path and expect them to
cost more.

Within the range, size is the axis that matters and fold and chain count are not: six folds, one
to three chains, and two structures with unresolved gaps all behave the same at the same token
count.

**Accepted designs have been measured at 288 tokens**, the PD-L1 example in the table below. One
576-token campaign has since run to the end, and it is the only one: a two-chain 387-residue
hIL2R target with a 146-residue binder, two trajectories, 249 gradient rounds at a median 47.20 s
(p10 46.95, p90 47.46), 3 h 39 m of wall clock, ten redesign candidates screened and refolded, and
**no design accepted**. It stopped on its trajectory cap, not on an error, and held 19.07 GB of
host memory at its peak. AICLK median 1350 MHz over 12,461 samples taken during the run, on a
p150a. The whole record is `perf/bgx_size/CAMP576.md`. Two trajectories cannot put a rate on anything, so treat the range as "the gradient loop
runs and completes" above 288, and the acceptance rate as measured at 288 only.

Binder length inside a campaign costs nothing extra to worry about. Every length BindCraft 2
draws against the 115-residue PD-L1 target fits: the draw range is 60 to 180 residues, three
roundings collapse it to five padded sizes (192, 224, 256, 288, 320 tokens), and the largest runs
six consecutive gradient steps with the resident DRAM line flat, peaking at 19.19 GB of 34.226 GB
in the backward with 12.68 GB still free. Measured on a p300c at AICLK 1350 median over 808
samples taken during the step.

Rounding up to the bucket is faster, not slower: the PD-L1 complex at 211 tokens costs 4.504 s on
the trunk forward and the same design padded to 224 costs 1.369 s, same card and same AICLK 1350
median.

All of the size numbers above are a p150a. A p300c is a different chip and its own numbers are
the ones in the two paragraphs before this.

### One chip of a Wormhole Galaxy stops at 512 tokens

A Wormhole Galaxy chip has 12 banks of 1,073,741,792 B, 12.885 GB against the p150a's 34.226 GB,
and its ceiling is its own: **512 tokens completes a gradient round and every axis above it
refuses in the Evoformer backward.** Measured on dev `.107` card 30 with the box's agent stopped
so the ladder held the chip alone, 2026-09-29, at AICLK 1000 MHz median over 109 samples taken
during the rung -- 1000 is this part's ceiling, not a throttled 1350.

| tokens | outcome | held at the refusal, of 12.885 GB |
|---|---|---|
| 320 to 512 | runs and completes | |
| 544 | refused, card full: 151.5 MB wanted, 128.6 MB free | 12.756 GB |
| 576 | refused, no contiguous run: 6.3 MB a bank wanted, largest block 1.4 MB | 12.735 GB |
| 608 | refused, card full: 189.3 MB wanted, 149.3 MB free | 12.736 GB |
| 640 | refused, no contiguous run: 0.4 MB a bank wanted, largest block 0.2 MB | 12.776 GB |

Every rung above 512 holds about 12.74 GB of the 12.885 GB before it stops, so whether the last
refusal reads "full" or "fragmented" is only which buffer happened to be next: the chip is
saturated either way.

**Why the ladder did not stop at the first refusal.** At 544 the fused triangle attention
declines every call -- the same L1 circular-buffer clash that makes 544 the axis to avoid on a
p150a -- so the composed path runs and holds a `[544, 4, 544, 544]` fp32 score tensor, 2.576 GB
the fold would not otherwise need. A single refusal at 544 could therefore have been that clash
rather than the top of the card, and the run says as much itself. It is not: **576 refuses with
the fused arm serving every call**, and 544 refuses again with the same numbers when re-run
alone. The clash follows the token axis and not the board, declining at 544 and 608 and serving
at 576 and 640 exactly as it does on a p150a.

So a Wormhole chip does **not** reach the p150a's 576, and nobody should predict either number
from bank geometry: the earlier guess from 12 GiB in 12 banks put this wall near 384-416 tokens,
and 384, 416, 448, 480 and 512 all run, in a flat 1:45 to 2:12 a rung.

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
