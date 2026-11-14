# The protein-only RFD4 condition families

Nine families, each generating a real design on Blackhole through `tt-bio design --model rfd4`,
each with a test that drives the CLI. Branch `wk/rfd4-cond-protein`, row `rfd4-cond-protein`.

Every run below is the full recipe: every step, every recycle, no shortened schedule. All nine
ran on qb2 chip 3 (p300c), two designs per invocation, AICLK sampled during the fold and
**median 1350 MHz on every one**, none flagged as a clock artifact. Raw record
`perf/rfd4/cond/runs.jsonl`, grades `perf/rfd4/cond/grades.json`, both on the branch.

| id | tokens | recipe | s/design @1350 | what the input fixes | fixed part reproduced | seed floor |
| -- | -----: | ------ | -------------: | -------------------- | --------------------: | ---------: |
| unconditional | 100 | thinking_medium | 66.9 | nothing | - | 15.68 A |
| binder | 225 | medium/front_loaded | 73.8 | target, internal coords | 0.171 A / 460 atoms | 19.40 A |
| antibody_cdr | 329 | high/front_loaded | 231.0 | antigen, internal coords | 0.295 A / 424 atoms | 33.09 A |
| nanobody_cdr | 224 | high/front_loaded | 80.0 | target + framework | 0.085 A / 400 atoms | 29.90 A |
| enzyme | 223 | medium/midpeak | 61.2 | catalytic site + ligands | 0.092 A / 22 atoms | 15.76 A |
| motif_scaffold | 82 | medium/midpeak | 25.7 | 3 motif residues | 0.082 A / 28 atoms | 9.19 A |
| partial_diffusion | 225 | medium/front_loaded | 33.6 | the parent design | 0.163 A / 460 atoms | 0.40 A |
| inverse_folding | 59 | medium/flat | 25.5 | the whole backbone | 0.687 A / 351 atoms | sequence, 0.339 identity |
| structure_prediction_folding | 138 | thinking_medium | 53.4 | the sequence only | 0.839 A CA (seed 0) | 10.33 A |

Test for all nine: `tests/test_rfd4_families.py`.

## How these are graded, and why not against the input file

A design is graded against **the atom array the featurizer produced**, not against the input CIF
re-read from disk. Several of these inputs are not structures in the design's numbering:
`motif_scaffold` ships only its 3 motif residues, numbered in their own frame, against an
82-residue design numbered from 0, and `unconditional` ships a length spec with no coordinates at
all. Matching those by residue id finds a handful of accidental collisions and reports the
distance between unrelated residues as if it were a motif deviation. It read 5.2 A for
motif_scaffold; the motif is actually reproduced at **0.018 A CA**. `rfd4.featurize` hands back
the array the model saw, in the design's own numbering, with the conditioning attached per atom,
so one rule covers all nine.

What an input *fixes* then decides what the number means, and this is per family, not global:

* **a spec pin** (`condition_coordinate_1_atom`, `condition_InternalCoordinate_1_atom`) is
  something the design has to give back. Bar 1.0 A; the worst of the six is 0.295 A.
* **the sampler** holds `--inverse_folding` and `--partial` inputs through `x_1`, with nothing
  annotated, so the input structure is the pin even though no condition names it.
* **a prediction target** is an input with coordinates and no pin of either kind. Folding is this
  one: its conditioning is sequence-only, so grading it against a 1 A "held" bar would be grading
  a prediction as a copy. It is reported per seed and does not gate the verdict.

The floor is two of our own designs at two seeds, which is the scale any paired number has to be
read against. Where everything is held there is no coordinate floor and the floor is the
sequence: inverse folding returns the same backbone to 3.5e-07 A and two different sequences
(0.339 identity), which is the family working.

Only the binder has a full upstream design to pair against, and it is rfd4-e2e's 128-token run,
not this 225-token spec: **3.315 A against a 5.378 A seed floor** (`state/rfd4/parity/e2e.json`).
`rfd4-ref` captured module goldens for the other families, not full upstream designs, so they are
graded on reproduced conditioning plus seed floor instead. Saying which is missing rather than
blocking on all ten.

## unconditional

100 residues, no conditioning of any kind. The input is a length spec and carries no coordinates,
so there is nothing to reproduce and the grade is the floor alone: two seeds land **15.68 A**
apart with 0.230 sequence identity. A family that returned the same structure twice here would be
broken; this one is generating.

## binder

PD-L1, 225 tokens. The target is held by `C_CRDGRP` as 460 atoms of internal coordinates, its
sequence by `C_SEQ` over 115 residues, and 6 hotspot atoms by `C_HOT`. The target comes back at
**0.171 A** over those 460 atoms while the 110 designed residues move 19.40 A between seeds.

This is the family rfd4-e2e already graded against upstream end to end at 128 tokens: 3.315 A
designed-chain CA against upstream's same-seed design, seed floor 5.378 A, target held to 0.086 A.

## antibody_cdr

Fab 5G9 against tissue factor, 329 tokens, the largest and slowest of the nine at 231.0 s/design.
274 residues pinned, 424 antigen atoms held, six CDRs left free. The antigen returns at
**0.295 A**. Sequence identity between seeds is 0.861, which is high only because the framework
is pinned and just the CDRs vary; the 223 free residues still move 33.09 A.

Upstream ships no paired-Fv design input, only VHH, so the spec is derived from the Fab complex
in the repo (`perf/rfd4/cond/make_antibody_spec.py`).

## nanobody_cdr

VHH caplacizumab against MZB1, 224 tokens. CDR1/2/3 conditioning is already annotated in the
input upstream ships. Target plus framework return at **0.085 A** over 400 atoms, the tightest of
the nine, and the 124 free residues move 29.90 A.

## enzyme

M-CSA entry M0040_13pk, 223 tokens. The conditioning is 26 tokens: the atomized ADP, MG and 3PG
ligands plus four catalytic residues, which the design carries as chains B/C/D (27 + 1 + 11
atoms). Held to **0.092 A**. The 180 scaffold residues are free and move 15.76 A between seeds at
0.056 sequence identity, the lowest of the nine.

The ligands here are conditioning input, not chemistry this row models; ligand families proper
belong to `rfd4-cond-chem`.

## motif_scaffold

1QJG indexed motif, 82 tokens, the cheapest family at 25.7 s/design. Three motif residues at
tokens 19, 45 and 69 carry `C_CRD` + `C_SEQ`; the model scaffolds around them. The motif comes
back at **0.082 A all-atom, 0.018 A on CA** — the conditioning reaches the model essentially
exactly. The 79 scaffold residues move 9.19 A between seeds.

## partial_diffusion

A binder design re-noised with `--partial 0.4167` and run on, 225 tokens, 33.6 s/design.

The number that matters here is not the floor but the restart. 0.4167 leaves 25 of 60 steps, and
the two modalities restart at very different noise levels because their schedules differ: step 35
is **t = 0.9700** on the backbone, whose schedule is log-shaped, and **t = 0.5034** on the local
latents, whose schedule is a power. The backbone restarts 3 percent noised. So 0.401 A between
seeds and 0.945 sequence identity is the setting doing what upstream's antibody-refinement setting
is for, and a 0.5 A floor bar would have failed a family that is working. A step fraction is not
a noise level, and only a per-modality `t` says honestly where a partial run started.
`perf/rfd4/cond/partial_start_t.py` prints it, card-free.

## inverse_folding

2KL8, 59 tokens, 25.5 s/design, `--tsr_k 2` at upstream's example temperature. The backbone is
held by the sampler through `x_1`, not by any condition, and comes back at **0.687 A all-atom and
3.5e-07 A on CA** — the two designs share a backbone bit for bit. Only the sequence is designed,
and the two seeds agree on 0.339 of it. There is no coordinate floor to quote for this family and
quoting one would be misleading.

## structure_prediction_folding

7HIY, 138 tokens, 53.4 s/design. Folding is inverse folding's mirror: `C_SEQ` on all 138
residues, no coordinate conditioning, so the model places a sequence it is told. Not in the
brief's eight, but `FEATURES.json` already assigned it to this row.

**The two seeds disagree, and this is the one open question in the set.** Seed 0 reproduces 7HIY
at **0.839 A CA** over 126 residues; seed 1 misses at **10.551 A**. Both returned the pinned
sequence exactly (identity 1.000), so the conditioning is reaching the model; what varies is
whether the fold converges. Two samples cannot separate the model's own spread from a port
defect, and there is no upstream 7HIY design to pair against, so it is recorded per seed rather
than called. This is what `rfd4-quality`'s distributional bar (>=30 designs, KS p>=0.05) exists
for, and folding should be the first family it is pointed at.

## A serving-path defect the card runs found

The first run on a card raised `IndexError: index 1 is out of bounds` in `write_cif` after
writing design 0: the writer takes a design row out of `features`, which hold one example and so
have one row, and upstream repeats the features per design before writing. Every family here
takes two designs in one invocation, so this would have cost all nine their seed floor.

The family tests passed before this, because they ran one design each, and no accuracy gate would
have caught it either: it is in the writer, past the model.
`tests/test_rfd4_families.py::test_two_designs_in_one_invocation_write_two_cifs` now drives the
CLI with `--num_designs 2`; it fails on the parent commit and passes on the fix (checked both
ways). Fixed in e6240569d.

## What is not here

No speed work: slow families are reported, not optimized (`antibody_cdr` at 231 s/design for 329
tokens is the one to look at, and it belongs to the `rfd4-perf-*` rows). No ligand, nucleic acid,
glycan or assembly families, which are `rfd4-cond-chem`'s. No beam search, no `evaluate`.
