# Docking with `tt-bio dock`

`tt-bio dock` places a small molecule in a receptor and ranks the poses. One model today,
DiffDock-L: diffusion over the ligand's position, orientation and torsion angles, with a
separate confidence model that scores and orders what came out.

```bash
tt-bio dock receptor.pdb --ligand 'CC(=O)Oc1ccccc1C(=O)O' --model diffdock
tt-bio dock ligands.csv --model diffdock --out_dir screen/      # a screen
tt-bio dock job.yaml --model diffdock                           # the tt-bio job form
```

The positional argument is a receptor structure (.pdb or .cif), a protein sequence, a CSV, a
job YAML, or a directory of YAMLs. Defaults are DiffDock-L's published recipe, so a plain
command reproduces the paper's inference settings: 10 poses, a 20-step schedule of which 19
run, no noise on the final step.

Both networks run on host torch today, so a dock needs no card. The score network is graded
against upstream's own at PCC 0.9999999999984 or better on every output, over three complexes
at five points of the schedule. `--no_model` still runs upstream's randomized seed-conformer
baseline, which needs no checkpoint.

One deliberate difference from upstream: where upstream keeps at most 32 neighbours per ligand
atom, this port keeps every neighbour inside the radius, because the subset upstream's cap keeps
changes when you rotate the same molecule and a pose should not depend on the frame its input
was written in. The cap binds on 1 of the 428 PoseBusters ligands, so almost no run sees a
difference at all, and a dock whose ligand does reach it warns, naming the complex, so you
never compare such a pose against an upstream number without knowing.

## Input

Upstream's CSV works unchanged, with the same four columns:

```csv
complex_name,protein_path,protein_sequence,ligand_description
1a46,receptors/1a46.pdb,,CC(=O)Oc1ccccc1C(=O)O
peptide,,MKTAYIAKQRQISFVKSHFSRQ,ligands/lig.sdf
```

An empty `complex_name` becomes `complex_<row>`; a row with a sequence instead of a path has
its receptor folded first. The YAML form is tt-bio's, and reads like every other model's:

```yaml
version: 1
sequences:
  - protein:
      id: A
      file: receptor.cif
  - ligand:
      id: L
      smiles: 'CC(=O)Oc1ccccc1C(=O)O'
docking:
  diffusion_samples: 10
  pocket: [12.4, 3.1, -8.7]
```

`docking:` carries exactly what the command line carries. Anything docking cannot honour, a
nucleic-acid chain or an affinity request for instance, is refused or noted by the same check
every fold goes through (see [model capabilities](model-capabilities.md)).

## Output

```
screen/1a46/rank1.sdf
screen/1a46/rank1_confidence-0.42.sdf
...
screen/1a46/rank10_confidence-3.17.sdf
screen/1a46/confidence.json
screen/failures.csv            # only if a row failed
```

The SDF names are upstream's, down to the two copies of rank 1, so a pipeline that parses
them keeps working. `confidence.json` is tt-bio's addition: rank, confidence and filename, so
you do not have to read floats out of filenames. With `--no_confidence` there is no
confidence model, the ranking is sample order and the files are plain `rank<N>.sdf`.

In a screen one bad ligand does not stop the run. The failures land in `failures.csv` with
the error for each, and `--limit_failures` (default 5) decides when to give up. A single
complex on the command line fails immediately instead, with the reason.

A pose whose coordinates go non-finite during sampling is kept at its last finite position,
logged with a warning and given a confidence of -1000, so it ranks last instead of above a
real pose. The complex itself only fails if every one of its poses does.

## Several cards and the fleet

A screen spreads across cards one complex per card at a time. `--devices 0,1,2,3` starts a
worker on each named card (default: every card on the host) and hands the complexes out as
cards free up; `--controller URL` submits the screen to a running `tt-bio controller` instead,
and `--owner` sets its fairness key there (see [multi-host](multi-host.md)). Either way the
results land under `--out_dir` in the layout above, the same files a one-card run writes. All
poses of a complex stay on one card, because the sampler draws its noise over the whole pose
batch. A single complex, or `--no_model`, runs in-process on one card.

On the fleet path every complex is attempted and each failure is a row in `failures.csv`;
`--limit_failures` applies to a local run only.

## Pocket docking

Without a pocket the whole receptor is searched. With one, poses start inside it:

```bash
tt-bio dock receptor.pdb --ligand lig.sdf --pocket 12.4,3.1,-8.7 --pocket_radius 5
```

## Weights

`tt-bio weights --download diffdock` fetches the v1.1 release (124 MiB), verifies its
sha256 and extracts both checkpoints; a run fetches it on first use. To point at your own
copy, `--checkpoint` and `--confidence_checkpoint` take either a `.pt` file or a directory in
upstream's layout, so an existing `workdir/v1.1/score_model` works as is.

## Options

`tt-bio dock --help` lists every flag with its default. The mapping from upstream's names:
`--samples_per_complex` is `--diffusion_samples`, `--inference_steps` is `--sampling_steps`,
`--batch_size` is `--max_parallel_samples` (poses scored in one device call, default 10; it
halves itself if the card refuses a batch, and the poses it produces do not change either
way), `--save_visualisation` is `--save_trajectory`,
`--loglevel` is `--log`, and `--old_score_model`/`--old_confidence_model` collapse into
`--weights {diffdock-l,diffdock}`. Only `diffdock-l` runs: the confidence network is the old
architecture either way (upstream's recipe sets `old_filtering_model: true` and the released
checkpoint is that network), but the pre-DiffDock-L *score* architecture is not implemented
and has no released weights, so `--weights diffdock` is refused instead of quietly running
DiffDock-L. The nine low-temperature sampling constants have no flags
of their own; they are a fitted set, and `--config` takes upstream's
`default_inference_args.yaml` verbatim if you want to change them.

The rest, by what they change:

- `--name` names a single complex and its output directory (default `complex_0`, as upstream).
- `--actual_steps` is how many of the `--sampling_steps` actually run; the recipe stops one
  early (19 of 20).
- `--initial_noise_std` sets the width of the starting translation prior in units of
  `tr_sigma_max` (upstream `--initial_noise_std_proportion`, default the fitted value).
- `--init_at_residue` starts the ligand at a random receptor residue instead of the receptor
  centre (upstream `--choose_residue`).
- `--sampler ode` integrates the probability-flow ODE instead of the default stochastic
  reverse diffusion, `sde` (upstream `--ode`).
- `--deterministic` takes the randomness out of the diffusion steps (upstream `--no_random`);
  with `--seed` a run then reproduces step for step.
- `--minimize gnina` post-minimizes the top pose with gnina, found on `PATH` or given with
  `--gnina_path`. The default is `none`.
