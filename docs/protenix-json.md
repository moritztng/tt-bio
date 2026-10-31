# Protenix and Mol-1 JSON input

`tt-bio predict` reads the job-list JSON that upstream Protenix and Mol-1 take (`mol pred -i jobs.json`), so an
input written for either runs unchanged on any Protenix-family model (`protenix-v1`, `protenix-v2`, `mol1`):

```bash
tt-bio predict jobs.json --model mol1
tt-bio predict jobs.json --model mol1 --use_seeds_in_json --use_template --use_rna_msa
```

Every entity type is read: `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand` (`CCD_` codes including
multi-component glycans such as `CCD_NAG_NAG`, SMILES, or `FILE_` paths to an .sdf, .mol, .mol2 or .pdb file), `ion`, plus
`count`, explicit chain `id`s, `modifications` and `covalent_bonds` (the older `left_*`/`right_*` keys too).

| upstream flag | tt-bio |
|---|---|
| `--use_msa` (on) | on by default; `--single_sequence` turns it off |
| `pairedMsaPath` / `unpairedMsaPath` | used as given; a chain without them is searched like any other input |
| `--use_rna_msa` | `--use_rna_msa` |
| `--use_template` and `templatesPath` (.a3m, .hhr, .json) | `--use_template`; needs `kalign` on PATH for .a3m/.hhr |
| `--use_seeds_in_json` and `modelSeeds` | `--use_seeds_in_json`; otherwise `--seed`, default 101 for JSON |
| `-c` / `-p` / `-e` | `--recycling_steps` / `--sampling_steps` / `--diffusion_samples`; the `mol1` defaults are upstream's 20 / 200 / 5 |

The features tt-bio builds from a JSON job are the ones Mol-1's own featurizer builds, entry for entry, checked by
`tests/test_mol1_features.py` against upstream goldens. Reference conformers (`ref_pos`) come from the checkpoint's
own CCD table, as upstream's do: `mol1` downloads Mol-1's `components.cif.rdkit_mol.pkl` (150 MB) once and keeps a
31 MB conformer table from it, while `protenix-v1`/`protenix-v2` keep the CCD ideal coordinates. A ligand given as
SMILES or a file gets a fresh RDKit conformer on both sides, so its `ref_pos` is not expected to match upstream's.
