# Mol-1

[Mol-1](https://github.com/molence/Mol-1) is Molence Bio's biomolecular structure model. It predicts the 3D structure
of complexes of proteins, DNA, RNA, small molecules and ions, with per-atom and per-chain confidences. It is derived
from Protenix and keeps its architecture, with a wider pair track and eight lighter trunk blocks in front, so tt-bio
runs it on the same Protenix engine as `protenix-v1` and `protenix-v2`.

```bash
tt-bio predict jobs.json --model mol1
tt-bio predict complex.yaml --model mol1
```

## Inputs

Upstream's job-list JSON is read as-is, the same file `mol pred -i jobs.json` takes: proteins, DNA, RNA, ligands by
CCD code, multi-component glycans, SMILES or a structure file, ions, modified residues, covalent bonds, explicit chain
ids and `modelSeeds`. tt-bio's YAML input works too. Field-by-field details are in
[`protenix-json.md`](protenix-json.md). Any CCD code in Mol-1's own Chemical Component Dictionary is accepted, so a
ligand or modification newer than the CCD library the other Protenix models read folds here as it does upstream.

Proteins use an MSA by default. Precomputed `pairedMsaPath` / `unpairedMsaPath` files are used as given, and any
protein chain without one is searched through the ColabFold API or a local database, as for every other MSA model
(see the README's MSA section). `--single_sequence` folds without one, like upstream's `--use_msa false`.
Templates (`--use_template`) and RNA MSAs (`--use_rna_msa`) are off by default, as upstream.
Searched template hits (`.a3m`, `.hhr`) are kept only if their PDB entry predates upstream's 2021-09-30 cut-off, so
the first such run downloads the PDB release-date table (13 MB) unless `$MOL_ROOT_DIR/common` already holds
upstream's copy. Hit structures come from `$MOL_ROOT_DIR/mmcif` or PDBe.

## Guiding a prediction (Mol-1-constraint)

Mol-1 can be told what you already know about the binding: which residues sit in the binding site, which residues
should touch each other, and which regions to keep clear. Add a `constraints` object to a job and the fold switches
to Molence's Mol-1-constraint weights, which download on first use.

```json
{
  "name": "complex",
  "sequences": [...],
  "constraints": {
    "site":      {"source_chains": ["A"], "partner_chains": ["B"], "residue_ids": [["A", 8], ["A", 12]]},
    "contact":   {"pairs": [[["A", 8], ["B", 15]]]},
    "exclusion": {"source_chains": ["A"], "partner_chains": ["C"], "residue_ids": [["A", 22]]}
  }
}
```

- **site**: the listed residues of the source chains should contact the partner chains (within 4.5 A, any heavy atom).
  Each residue needs to reach any one partner, not all of them.
- **contact**: each pair should end up within 4.5 A. Give residues as `pairs`, or, for a ligand, nucleic-acid or
  per-atom token, 0-based indices on the featurized token axis as `token_pairs`.
- **exclusion**: the listed residues should stay more than 6.0 A from the partner chains.

All three are optional and combine. Positions are 1-based indices into the full sequence you supplied, not PDB author
numbering, and must be protein residues on a source chain. The same object works in tt-bio's YAML input, one entry
per kind under `constraints:`. Conditioning guides the prediction, it does not guarantee the geometry: check the
output, and treat an unsatisfied constraint as information about the complex.

## Outputs

A JSON job writes upstream's layout, one directory per job and seed:

```text
<out_dir>/mol1_results_<input>/<job>/seed_<seed>/predictions/
    <job>_sample_0.cif
    <job>_summary_confidence_sample_0.json
    <job>_full_data_sample_0.json        # with --write_pae
```

Samples are ranked by `ranking_score`, 0 the best. The summary carries pLDDT, pTM, ipTM, gPDE, `has_clash`, the
ranking score and the per-chain and chain-pair matrices; `full_data` adds the token-pair PAE, PDE and contact
probabilities and the per-atom pLDDT. Key names and scales are upstream's, described in Mol-1's
[`docs/output_format.md`](https://github.com/molence/Mol-1/blob/main/docs/output_format.md). `results.json` beside it
holds one row per job for scripts and the platform.

The CIF is written by upstream's own writer, so it carries the same categories as Mol-1's (`_entry`, `_chem_comp`,
`_entity`, `_entity_poly`, `_entity_poly_seq`, `_struct_conn` for covalent bonds, and `_atom_site` with per-atom pLDDT
as the B-factor). The copies of one `sequences` entry share an entity, so DockQ, gemmi and other mmCIF readers see the
same entities upstream reports.

A system that cannot be predicted gets a report in `ERR/<job>.txt` next to `results.json`, and the run exits 1 when
every job failed, 2 when some did.

## Defaults

The defaults are upstream's: 20 recycles, 200 diffusion steps, 5 samples, seed 101 for a JSON job.

| `mol pred` | `tt-bio predict --model mol1` |
|---|---|
| `-i` | the input path |
| `-o` / `--out_dir` | `--out_dir` |
| `-s` | `--seed`; several seeds through `modelSeeds` and `--use_seeds_in_json` |
| `-c` / `-p` / `-e` | `--recycling_steps` / `--sampling_steps` / `--diffusion_samples` |
| `--load_checkpoint_path` | `--checkpoint` |
| `--use_msa false` | `--single_sequence` |
| `--use_template`, `--use_rna_msa`, `--use_seeds_in_json` | same names |
| `--need_atom_confidence` | `--write_pae` |
| `--device` | `--devices` (card ids); a card that is not there is an error |
| `-d`, `--trimul_kernel`, `--triatt_kernel`, `--enable_fusion`, `--enable_tf32`, chunking options | none: GPU execution choices. tt-bio's own precision options are `--fast` and `--diffusion_precision` |

`--fast` is graded for Mol-1: same recycles, steps and samples, with bfp8 pair-stack weights on Wormhole and the shared fast precision set on Blackhole. On the 791-token 9DBP complex it folds in 502 s instead of 596 s on a Wormhole chip and 195 s instead of 210 s on a Blackhole p150a. Across 11 post-cutoff complexes and 3 seeds it docks the same complexes as the default mode on both chips, with DockQ within 0.01.

## Other upstream commands

| upstream | tt-bio |
|---|---|
| `mol json` | `tt-bio json structure.cif` (or a .pdb, or a directory), with `--altloc`, `--assembly_id` and `--include_discont_poly_poly_bonds`. This is upstream's converter, so the JSON is identical to Mol-1's; the first run downloads the Chemical Component Dictionary (520 MB) |
| `mol msa jobs.json` / `mol msa seqs.fasta` | `tt-bio prep jobs.json` writes `jobs-update-msa.json` beside it with the MSA paths filled in; a FASTA gets one directory per record and `msa_index.json`. Same file names as upstream |
| `mol prep jobs.json` | `tt-bio prep jobs.json --use_template` (add `--use_rna_msa` for RNA chains) searches the MSAs, then templates with hmmsearch, and writes `jobs-final-updated.json` beside the input. The hmmer binaries and databases take upstream's option names (`--seqres_database_path` and so on) |
| `mol doctor` | `tt-bio preflight mol1`: are the weights here and intact, can this machine reach where they come from, and is kalign there for templates |
| first-run download, `MOL_ROOT_DIR` | the weights download on first use into tt-bio's cache (`tt-bio weights --download mol1` to fetch ahead); see [`weights.md`](weights.md) |

## Weights and licence

The checkpoint is fetched from a byte-identical Hugging Face mirror first and Molence's own bucket second, and is
checked against the sha256 of the official file either way. Mol-1's code is Apache-2.0, and Molence
answered on 2026-11-02 that the weights are Apache-2.0 too
([molence/Mol-1#5](https://github.com/molence/Mol-1/issues/5)). Mol-1-constraint is a second checkpoint under the
same licence, fetched from Molence's bucket and checked against its sha256; it is downloaded only for a job that
carries a `constraints` object.
