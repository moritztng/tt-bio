# What each model takes as input

TT-Bio's `predict` models all read the same YAML/FASTA input, but they do not all support the
same things. A model that cannot honour part of your input **refuses the fold** and names the
models that can, so nothing is quietly dropped: the failure mode this replaces is a fold that
accepted a ligand, ignored it, and reported success.

The one exception is `properties: affinity`, which only omits an extra output rather than
changing the structure. That prints a warning and the fold runs.

<!-- BEGIN CAPABILITY TABLE (generated: python3 -m tt_bio.capabilities) -->
| model | ligand | RNA | DNA | no protein chain | cyclic | modifications | templates | structure template | bond constraint | residue-residue bond | pocket/contact | affinity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `boltz2` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `esmfold2` | yes | yes | yes | refused | yes | yes | refused | refused | yes | yes | refused | ignored, warns |
| `esmfold2-fast` | yes | yes | yes | refused | yes | yes | refused | refused | yes | yes | refused | ignored, warns |
| `protenix-v1` | yes | yes | yes | yes | yes | yes | refused | refused | yes | yes | refused | ignored, warns |
| `protenix-v2` | yes | yes | yes | yes | yes | yes | yes | refused | yes | yes | refused | ignored, warns |
| `openfold3` | refused | yes | yes | yes | yes | yes | yes | refused | yes | refused | refused | ignored, warns |
| `openbind` | yes | yes | yes | yes | yes | yes | yes | refused | yes | refused | refused | ignored, warns |
| `opendde` | yes | refused | refused | yes | yes | yes | yes | refused | yes | yes | refused | ignored, warns |
| `opendde-abag` | yes | refused | refused | yes | yes | yes | yes | refused | yes | yes | refused | ignored, warns |
| `rf3` | yes | yes | yes | yes | yes | yes | refused | refused | yes | refused | refused | ignored, warns |
<!-- END CAPABILITY TABLE -->

`boltz2` is the fallback for anything the others refuse: it takes the whole input language.

`tt-bio affinity --model nesso1` reads the same file through its own parser and is not in the
matrix, because it returns a scalar and no coordinates. It answers `properties: affinity`,
takes protein and ligand chains, refuses a third entity type, and warns about everything else
it cannot read (`msa:`, `modifications:`, `cyclic:`, `templates:`, `constraints:`) rather than
refusing, so a Boltz-2 affinity yaml stays reusable. See
[`docs/nesso1.md`](nesso1.md).

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
      templates: path/to.npz      # precomputed alignment; boltz2 takes a cif block instead
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

An `X` in a protein sequence folds as an unknown residue (UNK). Any other letter outside the 20
standard amino acids, such as `U`, is refused: declare that residue under `modifications:` with
its CCD code (`SEC` for selenocysteine).

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
- **no protein chain** -- an input made only of RNA, DNA or ligands. ESMFold2 conditions its
  trunk on a protein language model, so it needs at least one protein chain.
- **cyclic** -- `cyclic: true` on a protein chain closes the backbone head to tail. Each model
  gets it the way its upstream expresses a ring: Boltz-2, RF3 and the OpenFold3 family wrap the
  relative position encoding, and Protenix, OpenDDE and ESMFold2, which have no such encoding,
  receive the closing amide bond (C of the last residue to N of the first). On those three, only a
  protein chain can be cyclic.
- **modifications** -- a non-canonical residue substituted at a position, by CCD code. Every
  model folds the modified chemistry.
- **templates** -- a precomputed template alignment per protein chain. There is no template
  *search*: you supply the file.
- **structure template** -- a top-level `templates:` block naming a cif/pdb file per chain, the
  Boltz-2 form. Only Boltz-2 reads it; the other template models take the alignment npz above.
- **bond constraint** -- a covalent bond between two named atoms where at least one end is on a
  ligand or a modified residue (a covalent inhibitor, a glycan).
- **residue-residue bond** -- a `bond` between two standard polymer residues, such as a
  disulfide. RF3 and the OpenFold3 family refuse it: both were trained with those bonds removed
  from their data, so neither can read one. Boltz-2, Protenix, OpenDDE and ESMFold2 take it.

A ligand atom in a `bond` is named the same way on every model: its element and its 1-based
count among that element's atoms in the SMILES string, hydrogens not counted. In
`C=CC(=O)N`, `C1` is the first carbon written, `O1` the oxygen and `N1` the nitrogen. A CCD
ligand uses the CCD's own atom names. Boltz-2 also accepts the names its output structures
carry, and refuses a name that would mean different atoms under the two schemes.
- **pocket/contact** -- a binding constraint. It needs a constraint embedder in the
  checkpoint, which only Boltz-2 has.
- **affinity** -- a predicted binding affinity for a named binder chain. Boltz-2 has the
  affinity head; `tt-bio affinity --model nesso1` predicts affinity without folding.

## Outputs

Every structure model writes a ranked `.cif`/`.pdb` per sample and per-atom pLDDT in the
B-factor column. Chains keep the ids you submitted, so a ligand sent as `L` comes back as `L`.
`--diffusion_samples N` draws N samples and writes all of them, best first. `--seed` makes a
run reproducible: two runs at the same seed give byte-identical files.

`--output_format pdb` has to fit the format's fixed columns, which mmCIF does not. The chain id
gets one column, so chain names longer than one character are rewritten `A`, `B`, `C`... in the
order the chains appear, and a residue name longer than three characters (a five-character CCD
code) is cut to three. Everything renamed is listed in a `REMARK 999` block at the top of the
file, so the names you submitted are still in it. A structure whose names already fit is written
unchanged and carries no remark. A PDB holds at most 62 chains; past that, use `cif`, which has no
width limit and never renames anything.

`--write_pae` adds a token-token PAE and PDE matrix as `<name>_pae.npz` on `boltz2`,
`protenix-v1`, `protenix-v2` and `opendde`. `rf3` writes pTM, ipTM and chain-pair PAE into
`<name>_summary_confidences.json` instead. `openfold3` and `openbind` compute PAE logits but
their fold does not return the matrices, and `esmfold2` has no PAE head.

An output flag a model does not read prints a note saying which model does read it, so it is
never silently accepted: `--write_pde` on Protenix (`--write_pae` already writes both) and
`--write_embeddings` outside Boltz-2.

`--max_msa_seqs` caps alignment depth on every model that folds from an MSA. Left alone it
changes nothing: Boltz-2 and ESMFold-2 keep their shipped 8192 default, and `protenix-v1`,
`protenix-v2`, `opendde`, `opendde-abag`, `rf3`, `openfold3` and `openbind` keep folding the
resolved alignment whole, which is the depth their reference numbers were measured at. Set it
and all of them cap. Every fold writes the depth it actually used as `msa_depth` in
`results.json`.

## Keeping this honest

The table is generated from `CAPABILITY` in `tt_bio/capabilities.py`, which is what the code
actually enforces, and `tests/test_capabilities_doc.py` fails if this file and that table
disagree. `tests/test_input_capabilities.py` runs the full model x feature cross product
against the real reader, so a verdict here is a measured behaviour rather than a claim.
