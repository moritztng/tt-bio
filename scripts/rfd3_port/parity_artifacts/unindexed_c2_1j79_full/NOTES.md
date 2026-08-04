# unindexed_C2_1j79 (full, WITH `ligand`) -- p20, SHIPPED

The real, unmodified `unindexed_C2_1j79` example from
`docs/examples/symmetry.md` (RosettaCommons/foundry, `models/rfd3`), verbatim
except adding the real, pre-existing `allow_ligand_on_existing_chain: true`
passthrough field (needed since 1j79's two ligand codes -- `ORO`/`ZN` -- land
on the same raw non-polymer chain, same situation as F4 enzyme design's
M0255 fixture).

Real input: `1j79_C2.pdb` (same file as `../symmetry_motif_1j79_nolig/`,
which is this same example MINUS `ligand` -- the variant p19 shipped and
value+device-trajectory parity verified).

## What p19 got right, and the one thing it couldn't root-cause

p19's exploration correctly established the real mechanism: 1j79's real
deposited PDB has ORO + 2x Zn in EACH of its two chains' active sites
(verified via `grep HETATM`), and the reference groups each subunit's own
ligand instances onto their own fresh chain, marking all of them
`sym_entity_id=FIXED_ENTITY_ID` (never resymmetrized -- same mechanism (c)
as an unindexed motif, see the module docstring). What blocked shipping was
the apparent CROSS-SUBUNIT-BLOCK ordering of the emitted ligand tokens:
subunit-B's `ORO` before subunit-A's, while subunit-A's two `ZN` before
subunit-B's -- not a consistent subunit-order or chain-letter-order rule.

## p20 root cause: it isn't a rule at all

`unravel_components` (`foundry/utils/components.py`) resolves a CCD code
with multiple physical matches via `components = list(set(components))` --
an UN-SORTED Python `set` over `f"{chain_id}{res_id}"` strings. Re-running
this exact capture (identical PDB/spec/seed) three times with three
different `PYTHONHASHSEED` values gives three DIFFERENT interleavings of
which subunit's atoms land first (verified: `asym_id` sequence for the 26
ligand atoms differed across all 3 runs). This is a real (if minor)
upstream reference bug -- unordered-set iteration where the surrounding
code assumes a stable order -- not a rule to reverse-engineer. What IS
stable across every run: each real ligand instance's own chain-of-origin,
`entity_id`/`sym_transform_id`/`sym_entity_id`/`is_sym_asu`, and the
PARTITION of atoms into per-real-residue `residue_index` groups (verified:
one subunit's ORO+Zn+Zn always land in 3 groups of sizes {11,1,1}, never
merged or split differently).

## What shipped (p20)

`tt_bio.rfd3_featurize._plan_ligand_tokens` groups ligand matches by their
real SOURCE chain (`residue.chain`) -- each source-chain group gets its own
fresh output chain (reduces to the pre-existing F4 "one shared chain" rule
when there's only one source chain). Ligand tokens get
`sym_transform_id=-1` (never resymmetrized). All of a spec's ligand chains
(however many) share ONE `entity_id` (a new override, mirroring
`sym_group_entity_id`'s "no real sequence, but still one group" pattern).
`residue_index`/`ref_space_uid` are keyed by real residue IDENTITY
(chain, res_id), not CCD code -- a related latent bug this fixture exposed
(the old code-keyed version was only coincidentally right while every prior
fixture had exactly one instance per code) -- fixed for both the symmetric
and non-symmetric cases. A `token_bonds` dict-collision bug for two
instances of the SAME code (the second instance's tokens silently
overwriting the first's in a plain per-code dict) was also fixed. Also
fixed while grounding this fixture: `select_fixed_atoms` for a ligand code
NOT in the dict defaults to FULLY FIXED (root-caused at
`input_parsing.py::apply_selections`'s `continue`-on-absent, leaving the
array's global `True` init untouched), not fully diffused -- the pre-p20
`none_()` default was untested since every earlier fixture explicitly
listed every ligand code; and `_ELEMENT_TO_ATOMIC_NUMBER` was missing `ZN`
(30), the first metal-ion ligand this port has grounded against a capture.

Verification methodology given the non-reproducible ligand-block order:
positional bit-exact comparison for the deterministic protein/unindexed-
motif tokens, IDENTITY-matched comparison (by nearest real 3D `motif_pos`,
side-independent since both featurizers compute `real_coord - COM`
identically) for the ligand sub-block -- see
`scripts/rfd3_port/parity_artifacts/parity_symmetry_ligand.py`. Zero
mismatches against all 3 differently-hash-seeded reference captures.
Device-trajectory: `scripts/rfd3_port/verify_trajectory_symmetry_motif.py
4 1j79_full`.

Reproduce the reference capture (CPU, no ckpt):
    /home/ttuser/rfd3_local_env/bin/python scripts/rfd3_port/capture_ref_f_spec.py \
        --pdb scripts/rfd3_port/parity_artifacts/unindexed_c2_1j79_full/1j79_C2.pdb \
        --spec_json scripts/rfd3_port/parity_artifacts/unindexed_c2_1j79_full/spec.json \
        --out_dir /tmp/ref_1j79_full_capture --seed 42
