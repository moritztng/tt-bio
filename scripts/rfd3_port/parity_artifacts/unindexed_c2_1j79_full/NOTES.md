# unindexed_C2_1j79 (full, WITH `ligand`) -- p19 exploration, NOT shipped

The real, unmodified `unindexed_C2_1j79` example from
`docs/examples/symmetry.md` (RosettaCommons/foundry, `models/rfd3`), verbatim
except adding the real, pre-existing `allow_ligand_on_existing_chain: true`
passthrough field (needed since 1j79's two ligand codes -- `ORO`/`ZN` -- land
on the same raw non-polymer chain, same situation as F4 enzyme design's
M0255 fixture).

Real input: `1j79_C2.pdb` (same file as `../symmetry_motif_1j79_nolig/`,
which is this same example MINUS `ligand` -- the variant p19 actually
shipped and value+device-trajectory parity verified).

## Why `ligand` + `symmetry` is NOT shipped this pass

A real local CPU capture of this exact spec (`rc-foundry[rfd3]`, no ckpt)
showed the reference DOES support this combination, and the general shape is
groundable: 1j79's real deposited PDB already contains ORO + 2x Zn in EACH of
its two chains' active sites (verified via `grep HETATM`), and the reference
groups each subunit's own ligand instances onto their own fresh chain,
marking all of them `sym_entity_id=FIXED_ENTITY_ID` (never resymmetrized --
same mechanism (c) as an unindexed motif, see the module docstring).

What could NOT be root-caused in the time available: the CROSS-SUBUNIT-BLOCK
ordering of the emitted ligand tokens. Verified in the real capture:
subunit-B's `ORO` instance is emitted BEFORE subunit-A's, while subunit-A's
TWO `ZN` instances are emitted before subunit-B's -- i.e. NOT a consistent
"subunit order" or "chain-letter order" rule across the two different CCD
codes. Shipping a token order this port could not explain would risk exactly
the "avoid everything to be safe, silently wrong" recurring bug class
(p17/p18 durable lessons) -- so `tt_bio.rfd3_featurize.featurize` raises
NotImplementedError for `spec.symmetry and spec.ligand` instead of guessing.

## For whoever picks this up next (p20+)

Reproduce the reference capture (CPU, no ckpt):
    /home/ttuser/rfd3_local_env/bin/python scripts/rfd3_port/capture_ref_f_spec.py \
        --pdb scripts/rfd3_port/parity_artifacts/unindexed_c2_1j79_full/1j79_C2.pdb \
        --spec_json scripts/rfd3_port/parity_artifacts/unindexed_c2_1j79_full/spec.json \
        --out_dir /tmp/ref_1j79_full_capture --seed 42

Then read `rfd3.inference.symmetry.symmetry_utils.make_symmetric_atom_array`'s
real ligand-handling branch (grep for where it groups ligand instances by
their source chain / subunit) to find the actual ordering rule, rather than
re-deriving it from the capture's output alone.
