# What each model takes as input

TT-Bio's `predict` models all read the same YAML/FASTA input, but they do not all support the
same things. A model that cannot honour part of your input **refuses the fold** and names the
models that can, so nothing is quietly dropped: the failure mode this replaces is a fold that
accepted a ligand, ignored it, and reported success.

The one exception is `properties: affinity`, which only omits an extra output rather than
changing the structure. That prints a warning and the fold runs.

<!-- BEGIN CAPABILITY TABLE (generated: python3 -m tt_bio.capabilities) -->
| model | ligand | RNA | DNA | cyclic | modifications | templates | bond constraint | pocket/contact | affinity |
|---|---|---|---|---|---|---|---|---|---|
| `boltz2` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `esmfold2` | yes | yes | yes | refused | yes | refused | refused | refused | ignored, warns |
| `esmfold2-fast` | yes | yes | yes | refused | yes | refused | refused | refused | ignored, warns |
| `protenix-v1` | yes | yes | yes | refused | refused | refused | yes | refused | ignored, warns |
| `protenix-v2` | yes | yes | yes | refused | refused | refused | yes | refused | ignored, warns |
| `openfold3` | refused | yes | yes | refused | refused | yes | refused | refused | ignored, warns |
| `openbind` | yes | yes | yes | refused | refused | yes | refused | refused | ignored, warns |
| `opendde` | yes | refused | refused | refused | refused | refused | yes | refused | ignored, warns |
| `opendde-abag` | yes | refused | refused | refused | refused | refused | yes | refused | ignored, warns |
| `rf3` | yes | yes | yes | refused | refused | refused | refused | refused | ignored, warns |
<!-- END CAPABILITY TABLE -->

`boltz2` is the fallback for anything the others refuse: it takes the whole input language.

## The input language

```yaml
version: 1
sequences:
  - protein:
      id: A                       # or [A, B] for identical copies
      sequence: MKTAYIAK...
      msa: path/to.a3m            # or `empty` to fold this chain single-sequence
      modifications:              # 1-indexed position in this chain's sequence
        - position: 5
          ccd: TPO
      templates: path/to.npz      # precomputed alignment, openfold3/openbind only
      cyclic: true
  - rna:   {id: R, sequence: GAUC}
  - dna:   {id: D, sequence: GATC}
  - ligand: {id: L, ccd: ATP}     # or a list of codes, or `smiles: c1ccccc1`
constraints:
  - bond: {atom1: [A, 5, SG], atom2: [L, 1, C1]}
  - pocket: {binder: L, contacts: [[A, 5]]}
properties:
  - affinity: {binder: L}
```

A FASTA gives the same chains without the keyed features: `>A|protein|msa.a3m`,
`>R|rna`, `>D|dna`, `>L|ccd` (the sequence line is the code) or `>L|smiles`.

Every key is checked when the file is read. A key outside this language is a typo and is
refused with the accepted set, because a dropped key used to cost a whole chain
(`protien:`) or a whole constraint block (`constrains:`) with no message at all.

## What each column means

- **ligand** -- a `ligand:` chain by CCD code or SMILES. `openfold3` refuses it on purpose:
  preview2 was released as a polymer model and was never trained to place a ligand, so it
  would return a confident structure for one anyway. `openbind` is the checkpoint upstream
  trained for co-folding, and it is the same implementation.
- **RNA / DNA** -- a nucleic-acid chain. `opendde` is protein and ligand only.
- **cyclic** -- `cyclic: true` on a polymer chain. Only Boltz-2 closes the backbone. Express
  the cyclisation as a covalent `bond` constraint on the models that take one.
- **modifications** -- a non-canonical residue substituted at a position. ESMFold2 folds the
  modified chemistry; the others have no featurizer argument for it.
- **templates** -- a precomputed template alignment per protein chain. There is no template
  *search*: you supply the file.
- **bond constraint** -- a covalent bond between two named atoms (a covalent inhibitor, a
  glycan, a crosslink). RF3 carries bonds through its own JSON/CIF spec, not through YAML.
- **pocket/contact** -- a binding constraint. It needs a constraint embedder in the
  checkpoint, which only Boltz-2 has.
- **affinity** -- a predicted binding affinity for a named binder chain. Boltz-2 has the
  affinity head; `tt-bio affinity --model nesso1` predicts affinity without folding.

## Outputs

Every structure model writes a ranked `.cif`/`.pdb` per sample and per-atom pLDDT in the
B-factor column. `--write_pae` adds a token-token PAE/PDE matrix on `protenix-v1`,
`protenix-v2` and `opendde`; `rf3` writes pTM, ipTM and chain-pair PAE into
`<name>_summary_confidences.json`; `openfold3`/`openbind` compute PAE logits but do not
return the matrices, and say so. `--diffusion_samples N` draws N samples and writes all of
them, best first. `--seed` makes a run reproducible.

## Keeping this honest

The table is generated from `CAPABILITY` in `tt_bio/capabilities.py`, which is what the code
actually enforces, and `tests/test_capabilities_doc.py` fails if this file and that table
disagree. `tests/test_input_capabilities.py` runs the full model x feature cross product
against the real reader, so a verdict here is a measured behaviour rather than a claim.
