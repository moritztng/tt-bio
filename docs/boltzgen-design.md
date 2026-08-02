# BoltzGen binder design

[BoltzGen](https://github.com/HannesStark/boltzgen) designs protein binders against a
target. The pipeline runs design → inverse folding → folding → analysis → filtering
and writes the top-ranked binders to `<out_dir>/final_ranked_designs/`.

```bash
tt-bio design examples/binder.yaml --model boltzgen --num_designs 10
```

This automatically uses every available card (splitting the designs across them and
merging the results) and writes to `./binder/`. Add `--devices 0,2` to run on
specific cards only. Weights download automatically on first use.

`boltzgen` is the default `--model`, so `--model boltzgen` may be omitted. The old
`tt-bio gen run ...` spelling still works as a deprecated alias and forwards every
flag unchanged.

## Input format

```yaml
entities:
  - protein:
      id: B
      sequence: 80..120         # designed chain, sampled length per design
  - file:
      path: target.cif          # target structure (path relative to this yaml)
      include:
        - chain:
            id: A
```

`80..120` randomises the binder length per design; a fixed integer pins it. Ligand,
DNA, and RNA targets use the same YAML grammar as `tt-bio predict`. See the
[BoltzGen examples](https://github.com/HannesStark/boltzgen/tree/main/example) for
binding sites, scaffolds, and residue constraints.

## Protocols

`--protocol` sets defaults appropriate for the binder type.

| Protocol | Use for |
|----------|---------|
| `protein-anything` (default) | de-novo protein binder |
| `peptide-anything` | peptide binder |
| `nanobody-anything` | nanobody / VHH |
| `antibody-anything` | antibody |
| `protein-small_molecule` | binder against a small-molecule target (adds affinity step) |
| `protein-redesign` | re-design existing residues (e.g. symmetric dimers) |

## Running a subset

`--steps` restricts the pipeline.

```bash
tt-bio design examples/binder.yaml --steps design --num_designs 10
tt-bio design examples/binder.yaml --out_dir existing/ --steps analysis filtering
```

## Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `--protocol` | `protein-anything` | Protocol; sets defaults appropriate for the binder type |
| `--num_designs` | `10000` | Number of binders to generate |
| `--budget` | `30` | Number of top designs kept after filtering |
| `--out_dir` | `./<basename>/` | Output directory |
| `--steps` | (all) | Run only specific stages |
| `--config STEP key=val` | — | Override per-stage config (e.g. `--config design sampling_steps=200`) |
| `--devices` | all cards | Restrict to specific cards (e.g. `0,2`) |
| `--fast` | `False` | Use a lower-precision path for some ops (slightly lower precision, faster) |
| `--cache` | `~/.boltz/boltzgen` | Cache for downloaded weights |
| `--debug` | `False` | Disable live display; show raw stage output |
| `--debug --log` | `False` | Add per-stage progress markers |
