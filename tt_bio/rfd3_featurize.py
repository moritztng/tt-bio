"""RFD3 host featurizer: build the ``f`` feature dict + initial token/pair state
metadata from a real user PDB/CIF + a parsed :class:`InputSpecification`.

This is the N6 core that turns a from-PDB design input into the ``f`` dict the
on-device TokenInitializer + DiffusionModule consume. It is grounded in the
real RosettaCommons/foundry featurizer
(``models/rfd3/src/rfd3/transforms/design_transforms.py`` + ``virtual_atoms.py``
+ ``pipelines.py``, production branch, 2026-07-23) and the ``f`` contract the
TokenInitializer reads.

Status (p15): ATOM-LEVEL parity landed for the protein-binder/motif-scaffold
case (F1/F6) AND the nucleic-acid-binder case (F2/F8). The reference does NOT
pad every token to a fixed 14 atoms —
``PadTokensWithVirtualAtoms`` only pads DESIGNED (sequence-unknown) tokens to
14; MOTIF (fixed-seq, indexed) tokens keep exactly their real observed heavy
atoms, looked up via the "dense" association scheme
(``rfd3.constants.association_schemes["dense"]``, vendored below as
``_DENSE_ATOM14_SCHEME``) which assigns each residue's real atoms to a
per-residue-type slot (with symmetry-reserved gaps, e.g. GLU's OE2 lands at
slot 9 not 8). Beyond backbone (N/CA/C/O/CB), atom NAMES are relabeled to
generic ``V0..V8`` for BOTH motif and designed atoms (``ATOM14_ATOM_NAMES``) —
this hides side-chain chemical identity from the atom-name channel while still
conditioning on real 3D geometry via ``motif_pos``. Verified against a local
CPU capture of the real reference featurizer (``rc-foundry[rfd3]``, no ckpt
needed): see ``scripts/rfd3_port/parity_artifacts/``.

Protein-specific reference-feature semantics (from
``CreateDesignReferenceFeatures.forward``, where ``has_sequence`` excludes
protein under ``generate_conformers_for_non_protein_only``): ``ref_pos``,
``ref_mask``, ``ref_pos_is_ground_truth``, ``ref_charge`` are all-zero/False
for EVERY protein atom (motif or designed) — real motif coordinates flow only
through ``motif_pos``. ``ref_element`` is likewise never filled for protein,
so its one-hot is the constant index-0 row for every atom (not real chemical
identity). ``motif_pos`` is centered: the whole design is translated so the
center of mass of the real (motif) atoms sits at the origin.

Coverage: protein-binder (F1) + motif-scaffolding (F6, indexed AND unindexed)
on protein input, nucleic-acid-binder design (F2/F8: a fixed-sequence DNA/RNA
target chain + a designed protein binder chain, e.g. the ``dsDNA_basic``/
``RNA_basic`` reference examples), small-molecule-binder design (F3: a real
ligand named by CCD code via the separate `ligand` spec field, e.g. the real
``sm_binder_design.md`` "buried"/"partial" examples), PLUS enzyme design (F4:
multiple ligand instances via a comma-separated `ligand` field, e.g.
``"NAI,ACT"``, plus `select_fixed_atoms`-subsetted catalytic protein residues
via `unindex`, e.g. the real ``enzyme_design.md``/M0255_1mg5 example — see the
F4 grounding below), PLUS cyclic/dihedral symmetric oligomer design (F5: the
``symmetry: {"id": "C3"}``-style field on a fully-unconditional (no `input`,
no `contig`, bare `length`) design spec, e.g. the real ``symmetry.md``
``uncond_C5``/``uncond_D4`` examples — see the F5 grounding below). NA as an
*indexed motif inside a protein chain*'s unindex field, the unindex numeric-
offset-tie syntax / dict-form per-atom fixing (see ``_plan_unindexed_tokens``),
and F5 symmetry COMBINED with any real input structure/motif/ligand all still
raise NotImplementedError with a pointer to the reference transform.

F5 symmetry grounding (verified against the real reference's own
``docs/examples/symmetry.md`` + ``rfd3.inference.symmetry.{symmetry_utils,
atom_array,frames}.py``, via a real local CPU capture of the doc's own
``uncond_C5``-shaped spec, e.g. ``{"length": 12, "is_non_loopy": true,
"symmetry": {"id": "C3"}}`` — no ``input`` PDB needed for this case, via
``capture_ref_f_uncond.py``):
- Scoped THIS PASS to the fully-unconditional case only: no `input` PDB, no
  `contig` (bare `length` only — verified the reference itself REQUIRES a
  real atom array the moment `contig` is set at all, even for an all-Designed
  contig string with zero Indexed components; only a bare `length` field
  skips that requirement), no `ligand`, no `unindex`. Symmetric MOTIF
  scaffolding (`is_symmetric_motif`, `is_unsym_motif`, the real
  ``unindexed_C2_*``/``indexed_unsym_C2_1bfr``/``unsym_C3_6t8h`` examples,
  which all combine symmetry with a real input structure) is NOT grounded
  this pass — raises NotImplementedError (see F5 durable-lesson note, p19+).
- Mechanism (`rfd3.inference.symmetry.symmetry_utils.make_symmetric_atom_array`):
  the parsed ASU (asymmetric-unit) token list — built by the SAME contig/
  designed-length machinery as every non-symmetric spec — is replicated
  `len(frames)` times, where `frames` is a list of `(R[3,3], t[3])` rigid
  transforms: `len(frames)==order` for `"C<order>"` (cyclic,
  `R = rotate 2*pi*i/order about z`, `t=0`), `len(frames)==2*order` for
  `"D<order>"` (dihedral: the same `order` cyclic rotations, PLUS each
  composed with a 180-degree in-plane flip — verified bit-exact against
  `rfd3.inference.symmetry.frames.get_cyclic_frames`/`get_dihedral_frames`).
  Copy 0 (`transform_id=0`, `R=I`) IS the original ASU, unchanged; each
  replica 1..N-1 gets its own fresh synthetic chain letter (same
  `_fresh_chain_letter` used elsewhere), each restarting `residue_index` at 0
  (verified: the reference's replicas are ordinary new chains, not a
  continuation of the ASU's chain).
- `entity_id`/`sym_id` special case: ALL replicas of one symmetric ASU share
  ONE `entity_id` (verified against a real capture: 3 C3 replicas of a
  12-residue unconditional design all get `entity_id=0`, NOT 3 distinct
  entities) with `sym_id` enumerating replicas 0..N-1 in ASU-then-replica
  order — this is a genuinely NEW case the pre-existing `entity_id` logic
  (grouping chains by matching real full-chain sequence, `chain_full_seq`)
  cannot handle on its own: a purely-DESIGNED (sequence-unknown) chain has no
  real sequence to match by content, so without an explicit symmetric-group
  override every replica would silently get its OWN fresh entity id (each
  designed chain looking "unique" by the pre-existing content-matching rule)
  — see `_symmetrize_tokens`/the entity_id computation below for the fix.
- Three NEW atom-level `f` keys beyond the existing 43 (verified via a real
  reference capture diffed against the identical spec with no `symmetry`
  field: `+sym_transform` [dict, NOT a plain tensor — `{str(transform_id):
  (R, t)}`, excluded from bit-exact tensor comparison the same way the
  reference's OWN capture script excludes it], `+sym_transform_id` [L] int32
  (which replica/transform an atom belongs to, broadcast from its token),
  `+sym_entity_id` [L] int64 (0 for every atom in this pass's single
  symmetric group — the reference's `FIXED_ENTITY_ID=-1` sentinel for
  non-symmetrized atoms is unreachable in this pass's no-ligand/no-motif
  scope), `+is_sym_asu` [L] bool (True only for `transform_id=0`'s atoms).
  All existing 43 keys' semantics are UNCHANGED (verified: entity_id
  special-case aside, every other per-atom/per-token feature is computed by
  the exact same code whether or not the token list happens to include
  symmetric replicas — a replica token is just an ordinary `_Token` on a new
  chain, not a new code path).
- Device-trajectory parity requires a matching SAMPLER-level mechanism, not
  just a featurizer one: the reference's `SampleDiffusionWithSymmetry`
  (`rfd3.model.inference_sampler`) re-derives every non-ASU replica's
  coordinates from the ASU's own (COM-recentered) denoised output at every
  step via `apply_symmetry_to_xyz_atomwise` (ported to
  :mod:`tt_bio.rfd3_sampler`) — it does NOT rely on the network alone to keep
  the design symmetric. Gated by `sym_step_frac` (default 0.9, unchanged from
  upstream): symmetrization is skipped once `c_t <= noise_schedule[floor(
  len(schedule)*sym_step_frac)]` (the last ~10% of steps, letting the network
  add final asymmetric detail) — for this port's own few-step device-
  trajectory checks (4-8 timesteps) this reduces to "every step except the
  last". `allow_realignment` (a separate, default-False upstream knob that
  additionally re-randomizes the whole structure's global pose every step) is
  OUT of scope this pass (upstream itself defaults it off; the real
  `uncond_C5`/`uncond_D4` example command never sets it).

F5 symmetry + a REAL motif (p19, grounded against real local CPU captures of
the real reference's own `unsym_C3_6t8h` example verbatim, PLUS a minimal
deterministic variant of `unindexed_C2_1j79` with its `ligand` field dropped
— the full `ligand`-bearing example is grounded and shipped as of p20, see
"ligand + symmetry" below — via `rfd3.inference.symmetry.{symmetry_utils,
atom_array,frames,checks,contigs}.py`):
- `get_symmetry_frames_from_atom_array` (mechanism (a)): when a real
  `structure_path` is given AND the built ASU has at least one real
  (fixed-coord) motif token AND `symmetry.is_symmetric_motif` is not
  explicitly `false` (the real default is `true`), the frames are NOT the
  closed-form `_cyclic_frames`/`_dihedral_frames` — they are derived from the
  RAW input structure's own real protein chains via a Kabsch alignment
  (`_symmetry_frames_from_structure`/`_kabsch_align`, a faithful numpy port
  of the reference's `_align`): group the raw structure's protein-only
  residues by chain, find the (single, this pass's scope) entity whose
  chain-count equals the symmetry id's own order (e.g. 2 identical chains
  for `C2`), sort those chains alphabetically, and Kabsch-align every
  non-reference chain's real backbone+sidechain coordinates onto the first
  (reference) chain's — the resulting rotation (translation always dropped,
  `t=[0,0,0]`, matching the reference) is frame `transform_id=i` for the
  `i`-th chain in that sorted order. This REQUIRES the same one-time
  pre-centering the reference applies before frame derivation
  (`center_symmetric_src_atom_array`, ported as
  `_center_symmetric_src_residues`): shift EVERY real residue's real
  coordinates (protein AND any motif/ligand) by the negative mean of
  PROTEIN-only real atoms in the RAW structure, once, before any token
  planning — otherwise a pure rotation (no added translation) cannot
  reproduce a real subunit's true position from another's. Verified
  bit-exact-to-float-precision against a real capture's `motif_pos` for a
  REAL, geometrically-transformed replica atom (the `unindexed_C2_1j79`
  variant's replica copy of unindexed residue A250 — see p19 VALUE-PARITY
  section). Scoped this pass to a single real symmetric protein entity whose
  chain count matches the symmetry id's order exactly (every real reference
  example this port grounds against is exactly this shape); a genuinely
  heteromeric or mismatched-multiplicity symmetric input raises
  NotImplementedError rather than guess.
- `is_unsym_motif` (mechanism (b), `rfd3.inference.symmetry.contigs.
  get_unsym_motif_mask`/`expand_contig_unsym_motif`): a comma-separated list
  on `symmetry.is_unsym_motif` naming contig/ligand components that must NOT
  be symmetrized — a bare token (e.g. `"HEM"`) matches a ligand CCD code (out
  of scope this pass, ligand+symmetry is blocked, see below) or a single
  `{chain}{res_id}`; a `-`-range (e.g. `"Y1-11"`) expands to individual
  `{chain}{res_id}` names, matching indexed motif RESIDUES already present in
  the built token list (verified against a real capture of `unsym_C3_6t8h`:
  the DNA contig components `Y1-11`/`Z16-25` are excluded from the 3x
  cyclic replication entirely — ONE copy each, physically placed AFTER all 3
  replicated protein copies, in their original relative contig order,
  keeping their OWN real chain letters unchanged). Verified: excluded
  (`is_unsym_motif`) tokens need NO special-case `entity_id` override — the
  PRE-EXISTING generic `chain_full_seq`-based entity grouping already gives
  Y and Z their own distinct entity ids correctly (they are genuinely
  different real DNA sequences), unlike the F5 "symmetric replica" special
  case (p18's `sym_chains` override), which is a DIFFERENT mechanism only
  needed for tokens that ARE replicated.
- Unindexed-motif replication ("mechanism (c)", verified against the
  `unindexed_C2_1j79`-minus-`ligand` variant): an unindexed motif token NOT
  flagged `is_unsym_motif` (e.g. `unindex: "A250"`, a catalytic residue
  "within a subunit") stays in the replicated ASU stream exactly like an
  ordinary protein token — it gets `len(frames)` real, geometrically-correct
  copies (one per subunit, each the ASU's own real atom coordinates rotated
  by that subunit's frame) — its replica copies share the SAME per-replica
  chain letter as the main protein/contig replicas (reusing one
  `replica_chains` list across both `_symmetrize_tokens` calls, needed so a
  replica's unindexed-motif copy and its subunit's main chain share one
  `asym_id`/`entity_id`, verified against the real capture). AFTER
  replication, EVERY unindexed-motif atom (regardless of which replica it
  came from) is forced back to the F5 "fixed" sentinel
  (`sym_transform_id=-1`, `sym_entity_id=-1`, `is_sym_asu=False`) — a direct
  port of `fix_3D_sym_motif_annotations`'s post-hoc override
  (`_is_motif & ~_is_indexed_motif`) — so the SAMPLER's existing generic
  `apply_symmetry_atomwise` (unchanged, already skips any `sym_entity_id ==
  FIXED_ENTITY_ID` atom) never overwrites these already-geometrically-placed
  replica copies with a bogus reconstruction. This composes with the
  PRE-EXISTING, symmetry-independent "never add noise to a fixed-coord atom"
  sampler mechanism (unindexed motif atoms are always `is_motif_atom_with_
  fixed_coord=True` for protein, see F4 grounding) — the two mechanisms are
  redundant-but-consistent by design, exactly like the real reference's own
  belt-and-suspenders annotation.
- **`ligand` + `symmetry` (p20, root-caused and shipped)**: grounded against
  the REAL (unmodified) `unindexed_C2_1j79` example — the reference does NOT
  treat a symmetric design's ligand as one excluded, un-replicated instance
  the way F3/F4's original single-instance model assumed. A genuinely
  symmetric input PDB's ligand is typically ALREADY physically duplicated
  once per real subunit in the deposited coordinates (1j79's real PDB has an
  ORO + TWO Zn ions in EACH of its two chains' active sites, verified via
  `grep HETATM`) — the reference's `_append_ligand` picks up EVERY real
  instance matching the requested CCD code(s) (needs
  `allow_ligand_on_existing_chain: true`, a real passthrough field), groups
  them by which real input chain (subunit) they physically belong to, gives
  each subunit-group its OWN fresh chain (`_plan_ligand_tokens` groups by
  `residue.chain`, generalizing the pre-existing "multiple codes share one
  chain" F4 rule — a non-symmetric input's ligands all share one real source
  chain, so this reduces to exactly the old F4 behavior with zero regression;
  a symmetric input's ligands, physically duplicated per subunit, split into
  one fresh chain PER subunit), and marks ALL of them
  `sym_entity_id=FIXED_ENTITY_ID` (matching mechanism (c) above — a ligand,
  symmetrized input or not, is NEVER resymmetrized by the sampler:
  `sym_transform_id=-1` for every ligand token). A further real finding,
  needed for the general `entity_id` feature (separate from the F5-specific
  `sym_entity_id`): every ligand chain — however many subunit-groups it was
  split into — shares ONE `entity_id`, mirroring `sym_group_entity_id`'s
  "no real sequence, but still one group" override (verified: the two
  per-subunit ligand chains of the symmetric enzyme both land on the same
  entity, not two distinct ones; for a single-chain non-symmetric ligand this
  reduces to the pre-existing "give it a fresh entity" behavior, zero
  regression).

  **The CROSS-SUBUNIT-BLOCK ordering of the final ligand token block is NOT a
  rule at all — it is PYTHONHASHSEED-dependent and non-reproducible even by
  the reference itself.** `unravel_components` (`foundry/utils/components.py`)
  resolves a CCD code with multiple physical matches via
  `components = list(set(components)); return components` — an un-sorted
  Python `set` over `f"{chain_id}{res_id}"` strings. Re-running the IDENTICAL
  capture (`capture_ref_f_spec.py`, same PDB/spec/seed) three times with three
  different `PYTHONHASHSEED` values gave three DIFFERENT interleavings of
  which subunit's `ORO`/`ZN` atoms land first in the emitted token block
  (verified: `asym_id` sequence for the 26 ligand atoms differed across all 3
  runs). Everything downstream of that raw order (which specific
  `residue_index` slot / array position a given physical instance lands at)
  is consequently also run-dependent — this is a real, if minor, upstream
  reference bug (unordered-set iteration used where the surrounding code
  assumes a stable order), not a deliberate rule to reverse-engineer. What
  IS stable across every run (verified all 3): each real ligand instance's
  own chain-of-origin, its `entity_id`/`sym_transform_id`/`sym_entity_id`/
  `is_sym_asu`, and the PARTITION of atoms into per-real-residue
  `residue_index` groups (e.g. one subunit's `ORO`+`ZN`+`ZN` always land in 3
  groups of sizes {11,1,1}, never merged or split differently) — this port
  reproduces exactly that stable structure with its own deterministic order
  (codes in `spec.ligand`'s given order; multiple instances of one code in
  their structure-file order) rather than attempting to replicate an
  accident of CPython's per-process string-hash seed. Parity for this
  fixture is therefore verified by IDENTITY-matched comparison (per real
  `(chain, res_id, atom_name)`), not raw positional bit-exactness, for the
  ligand sub-block specifically — see
  `scripts/rfd3_port/parity_artifacts/unindexed_c2_1j79_full/` for the
  fixture, the multi-run evidence, and the verification script.

  The one remaining narrower gap: `is_unsym_motif` naming a `ligand` CCD code
  directly (rather than relying on the reference's own mechanism (c), which
  already treats every ligand as implicitly unsym) is still out of scope —
  `_plan_ligand_tokens`'s output never passes through the
  `is_unsym_motif`/`_token_matches_unsym` split at all (ligand tokens are
  appended after that split runs), so this specific combination still raises
  NotImplementedError explicitly rather than silently no-op.

Enzyme (F4) grounding (verified against a real local CPU capture of the real
``enzyme_design.md`` example — ``M0255_1mg5.pdb`` + its own
``ligand: "NAI,ACT", unindex: "A108,A139,A152,A156", select_fixed_atoms: {...}``
spec, via ``capture_ref_f_spec.py``; needed
``allow_ligand_on_existing_chain: true`` since this PDB's two hetero groups
land on the same raw chain):
- Multiple *different* ligand CCD codes in one ``ligand`` field (comma-
  separated, e.g. ``"NAI,ACT"``) all land on ONE SHARED fresh chain (verified:
  both NAI's and ACT's tokens get the SAME ``asym_id``/entity) — NOT one fresh
  chain per code as a single-ligand spec's chain allocation might suggest.
  ``residue_index`` increments once PER LIGAND INSTANCE on that shared chain
  (0 for the first code, 1 for the second, ...), matching how a real multi-
  atom residue occupies one `residue_index` slot regardless of atom count.
  **Corrected p20**: the real rule is one slot per REAL RESIDUE INSTANCE,
  keyed by that instance's own real (chain, res_id) — NOT one slot per CODE
  (this port's `(chain, res_name)`-keyed implementation through p19 was only
  coincidentally right, since every case verified through p19 had exactly one
  instance per code — the multi-instance case itself was still explicitly
  `NotImplementedError`-guarded, so this was a scoped gap, not a silent
  latent bug). 1j79's per-subunit ORO+2×Zn (see the F5+ligand grounding
  above) needed the fix: it always partitions into 3 `residue_index` groups
  of sizes {11,1,1}, never merged by code. Multiple instances of the SAME
  code are now supported (p20), keyed by real residue identity.
- ``ref_space_uid`` is a GLOBAL count of distinct residue-groups in first-
  appearance order (biotite's residue-level indexing) — NOT literally "this
  ligand's own first token index" (that framing, used through p16, happens to
  coincide for a single trailing ligand block but breaks for a second ligand
  instance: verified NAI's 44 atoms -> ref_space_uid 197 (correct, matches its
  first-token-index by coincidence) but ACT's 4 atoms -> ref_space_uid 198,
  NOT 241 (its real first-token index) — ACT is simply the 199th distinct
  residue-group, not token #241). Same p20 correction as `residue_index`
  above: the group key is the real residue INSTANCE `(chain, res_id)`, not
  the code — verified against 1j79's per-subunit ORO+2×Zn giving 6 distinct
  `ref_space_uid` groups among the 26 ligand atoms (sizes {11,11,1,1,1,1}),
  not 2 (one per code).
- ``select_fixed_atoms`` can SUBSET which of an UNINDEXED protein residue's
  real atoms enter the token at all (not merely flag them) — verified against
  the reference source (``input_parsing.py::_build_init``:
  ``unindexed_tokens[k] = tok[tok.is_motif_atom_with_fixed_coord]`` runs
  BEFORE tokenization, permanently dropping every other real atom). This is
  INDEXED (contig) motif-only exempt: an indexed/contig motif token always
  keeps every real atom regardless of `select_fixed_atoms` (verified: the
  subsetting call only ever touches `unindexed_tokens`, never
  `indexed_tokens`) — a dict-form `select_fixed_atoms` targeting an INDEXED
  motif residue is therefore still out of scope (NotImplementedError; would
  need genuine per-atom partial-fixing bookkeeping on a still-full atom set,
  a different mechanism than unindexed atom-subsetting).
- When an unindexed residue's `select_fixed_atoms`-kept atom subset has no
  real CB (or no CA, for glycine), the reference falls back to forcing the
  FIRST kept real atom to be its own representative (`add_representative_atom`
  -> per-atom `atomize`) — is_backbone/is_central True, is_sidechain False for
  that one atom, same convention already used for ligand atoms — verified
  against the real capture (e.g. A108 subsetted to ``{ND2,CG}``: neither
  survives as CB, so CG — the first KEPT real atom in input-structure order —
  becomes the representative, NOT literally "whichever real atom is CB").
- **A ligand CCD code with NO entry in `select_fixed_atoms` defaults to
  FULLY FIXED, not fully diffused (p20 fix)** — root-caused at the reference
  source (`input_parsing.py::_assign_types_to_input.apply_selections`): the
  WHOLE array's `is_motif_atom_with_fixed_coord` starts at the GLOBAL init
  value `True` (`REQUIRED_CONDITIONING_ANNOTATION_VALUES`); a per-residue
  `apply_selections` call `continue`s (leaves that residue's annotation
  UNTOUCHED) whenever `selection.get(f"{chain_id}{res_id}")` is `None` — i.e.
  a residue absent from `select_fixed_atoms` keeps the default `True`, it is
  NOT reset to `False`. Every fixture through p19 happened to explicitly
  list EVERY ligand code in `select_fixed_atoms` (even as `""`, meaning
  "fix nothing" — a real, deliberate per-code OVERRIDE, still correctly
  `none_()`), so "code entirely absent from the dict" was never exercised
  until `unindexed_C2_1j79`'s real ORO/Zn (present in the ligand field, but
  never named in that spec's `select_fixed_atoms: {"A250": ...}`) — verified
  they are ALL fixed (`is_motif_atom_with_fixed_coord=True` for all 26
  atoms) in a real reference capture, not diffused. This is a DIFFERENT
  default than `select_buried`/`select_exposed` (whose own global init is
  effectively "no label", so their existing `none_()`-when-absent default is
  correct and unchanged) — `_resolve_ligand_atom_selection` now takes a
  per-FIELD `not_selected` fallback, `all_()` for `select_fixed_atoms` only.

Ligand (F3) grounding (same design_transforms.py/virtual_atoms.py + the real
``rfd3.inference.input_parsing.py``/``rfd3.inference.parsing.py`` select-field
resolution, verified against a real local CPU capture of IAI.pdb/the
``sm_binder_design.md`` "buried" example via ``capture_ref_f_spec.py`` — no
ckpt needed, same method as F2/F8):
- A ligand is ATOMIZED: each real heavy atom is its OWN token (not grouped
  into one multi-atom token like protein/NA) — verified: ``PadTokensWithVirtualAtoms``'s
  ``is_residue`` gate (``is_protein & ~atomize``) excludes a ligand entirely,
  so it's never padded/grouped. Each atomized token's single atom is trivially
  its own representative: ``is_ca``/``is_central``/``is_backbone`` are all
  True and ``is_sidechain`` False for every ligand atom (verified).
- A ligand ALWAYS has known chemical identity (``is_motif``/"class 1", same
  as an indexed motif) even when its COORDINATE is diffused (unfixed) — the
  reference's ``select_unfixed_sequence`` field explicitly excludes ligands
  ("ligands / DNA always have fixed sequence"); `is_fixed_coord` and
  `is_fixed_seq` are therefore tracked as SEPARATE flags on `_Token` (unlike
  protein/NA, where they always coincide) — see the `_Token.fixed_coord`/
  `fixed_seq` properties and the ``select_fixed_atoms`` resolution below.
- ``ref_pos``/``ref_element``/``ref_charge``/intra-ligand ``token_bonds`` come
  from a REAL CCD template — this port reuses its OWN existing bundled CCD
  rdkit-mol library (``tt_bio.data.mol.load_molecules``, ``~/.boltz/mols``,
  the same one Boltz-2/Protenix-v2 already ship) rather than re-vendoring
  RDKit/CCD conformer generation. ``ref_pos`` is reference-CONFORMER geometry,
  not identity: the real reference itself draws a fresh, unseeded random
  RDKit ETKDG conformer + random rigid augmentation every run (verified in
  the reference source), so no single captured reference run's `ref_pos` is
  "the" bit-exact target — this port's OWN Protenix-v2 host featurizer
  already documents and relies on exactly this invariance
  (``tt_bio/protenix_data.py:466``).
- ``ref_atom_name_chars`` is overridden to encode the ELEMENT symbol, not the
  real atom name, for every ligand atom (the reference's
  ``use_element_for_atom_names_of_atomized_tokens=True`` default — verified: a
  real capture's ligand rows decode to "C   "/"N   ", not "C22 "/"N9  ").
- ``select_buried``/``select_exposed`` become the ``ref_atomwise_rasa`` one-hot
  bin DIRECTLY (0=buried, 2=exposed) — a user-specified per-atom LABEL at
  inference, not a computed SASA value (the real Shrake-Rupley RASA transform
  is training-only, never invoked at inference — verified in the reference
  source). A ligand-code dict key (e.g. ``{"IAI": "C1,C2"}``) is a SEPARATE
  convention from protein/NA's ``{chain}{res_id}`` key.
- ``restype`` for a ligand token is the protein-UNK slot (index 20, NOT the
  GAP/no-sequence slot 31) — verified against a real capture.
- ``residue_index``/``ref_space_uid`` are RESIDUE-level, not token-level: all
  of a ligand's atomized tokens share ONE residue_index (0, on the ligand's
  own fresh chain) and ONE ref_space_uid (the first ligand token's index) —
  verified against a real capture (getting this wrong assigns each ligand
  atom its own distinct "residue", silently misconditioning symmetry/pair
  features that key off `ref_space_uid`).
- A pure ``length``-only spec (no ``contig`` at all, e.g. a small-molecule
  binder with nothing but a fresh designed chain + a ``ligand``) is treated
  as a bare designed-length contig string (``parse_contig`` already parses a
  bare "180-180"/"180" as Designed/DesignedRange).
- Multiple instances of the SAME CCD code are supported (p20, see F4/F5+
  ligand grounding above for the residue_index/ref_space_uid fix this
  needed) — as is a comma-separated list of *different* codes (F4). Still
  scoped out: the ``TIP``/``BKBN`` atom-selection shorthands applied to a
  ligand, and a contig-string (rather than dict-form) select_* value
  targeting a ligand — both raise NotImplementedError rather than guess.

NA (F2/F8) grounding (``rfd3.transforms.design_transforms.py`` +
``virtual_atoms.py`` + ``util_transforms.py``, verified against real local CPU
captures of ``1bna.pdb``/dsDNA and ``1q75.pdb``/RNA via ``capture_ref_f.py`` —
no ckpt needed, same method as F1/F6):
- DNA/RNA atoms are NEVER renamed to generic ``V0..V8`` labels and NEVER
  padded — ``PadTokensWithVirtualAtoms``'s ``is_residue`` gate is
  ``is_protein & ~atomize`` (plus unindexed, N/A here), so non-protein tokens
  never enter that transform at all. Real atom names (``O5'``, ``C1'``,
  ``N9``, ...) are kept verbatim, in the input structure's real order — no
  scheme lookup needed (unlike the protein "dense" scheme).
- ``ref_element`` IS filled for NA (unlike protein, whose ``has_sequence``
  is unconditionally excluded by ``generate_conformers_for_non_protein_only``)
  — one-hot atomic number of the real parsed element. ``ref_charge`` is 0 and
  ``ref_mask`` is True for every NA atom (verified against both a real capture
  AND the persisted p4/p10 dsDNA_basic ckpt golden).
- ``ref_pos`` (the reference-conformer 3D geometry) is NOT reproduced this
  pass — the real pipeline calls into RDKit/CCD-template conformer generation
  (``get_af3_reference_molecule_features``), which this port does not vendor;
  left at 0 (documented gap, same simplification protein already uses since
  fixed atoms get real geometry via ``motif_pos`` regardless).
- ``is_ca``/``is_central`` (the one "representative atom" per token) is the
  base's ring-center atom: ``C4`` for purines (DA/DG/A/G), ``C2`` for
  pyrimidines (DC/DT/C/U) — verified against both captures.
- ``is_backbone``/``is_sidechain`` are never set for NA (that split is
  protein-only in the reference); ``terminus_type`` (5'/3') is likewise never
  set for NA in this contract (verified all-zero on both captures).
- ``restype``'s 32-dim one-hot follows the real AF3 vocabulary
  (``atomworks.ml.encoding_definitions.AF3_TOKENS``): 0-19 the 20 AA, 20
  unknown-AA, 21-24 RNA A/C/G/U, 25 unknown-RNA, 26-29 DNA DA/DC/DG/DT, 30
  unknown-DNA, 31 GAP (designed/no-sequence) — verified index-for-index
  against both captures.
- ``entity_id``/``sym_id``: chains are grouped into the same entity by their
  FULL real-chain residue-name sequence (not just the contig-selected
  subset) — e.g. dsDNA_basic's chain A and chain B are the same 12-mer
  palindrome, so they share ``entity_id`` and get distinct ``sym_id`` replica
  indices. A synthetic (designed) chain always starts a fresh entity. A
  ``Designed``/``DesignedRange`` segment immediately after a contig chain
  break (``/0``) gets a brand-new synthetic chain letter rather than
  inheriting the preceding indexed block's chain (verified: dsDNA_basic's
  designed protein segment is its own chain/entity, not chain "B").
"""
from __future__ import annotations

import copy
import os as _os
import re
from dataclasses import dataclass, replace as _dc_replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .rfd3_input import (
    InputSpecification, parse_contig, ChainBreak, Indexed, Designed, DesignedRange,
    AtomSelection, _parse_atom_spec,
)

# -- atom14 generic name template (rfd3.constants.ATOM14_ATOM_NAMES) --------
# Slots 0..4 keep real backbone/CB names; slots 5..13 are always the generic
# "V{i}" placeholder, whether the atom is a real (renamed) side-chain atom of
# a motif residue or a synthetic virtual pad atom of a designed residue.
ATOM14_ATOM_NAMES = ["N", "CA", "C", "O", "CB"] + [f"V{i}" for i in range(9)]
BACKBONE_NAMES = {"N", "CA", "C", "O"}

# The "dense" association scheme (rfd3.constants.association_schemes["dense"],
# stripped variant used by map_to_association_scheme): for each residue type,
# the REAL atom name occupying each of the 14 atom14 slots (None = unused
# slot for that residue — note the gaps, e.g. GLU's OE2 sits at slot 9, not
# 8, reserved for symmetry-consistent packing across residue types). A real
# atom's generic name is ATOM14_ATOM_NAMES[slot]; a motif residue emits only
# the slots that are both non-None here AND actually present in the input
# structure (no padding — this is what makes L variable per token).
_DENSE_ATOM14_SCHEME: dict[str, list[str | None]] = {
    "ALA": ["N", "CA", "C", "O", "CB", None, None, None, None, None, None, None, None, None],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", None, None, None],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", None, None, None, None, None, None],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", None, "OD2", None, None, None, None, None],
    "CYS": ["N", "CA", "C", "O", "CB", None, "SG", None, None, None, None, None, None, None],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", None, None, None, None, None],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", None, "OE2", None, None, None, None],
    "GLY": ["N", "CA", "C", "O", None, None, None, None, None, None, None, None, None, None],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", None, None, None, None],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", None, None, None, None, None, None],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", None, None, None, None, None, None],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", None, None, None, None, None],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", None, None, None, None, None, None],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", None, None, None],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", None, None, None, None, None, None, None],
    "SER": ["N", "CA", "C", "O", "CB", "OG", None, None, None, None, None, None, None, None],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", None, None, None, None, None, None, None],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", None, None],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", None, None, None, None, None, None, None],
}

# Real AF3 sequence vocabulary (atomworks.ml.encoding_definitions.AF3_TOKENS):
# 20 AA + unknown-AA, 4 RNA + unknown-RNA, 4 DNA + unknown-DNA, GAP. restype
# is a 32-dim one-hot over this exact order (index-verified vs real local
# captures of dsDNA_basic-style and RNA_basic-style inputs, see module
# docstring). Designed (sequence-unknown) tokens use the GAP slot (31).
_RESTYPE_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
    "A", "C", "G", "U", "N",
    "DA", "DC", "DG", "DT", "DN",
    "GAP",
]
_RESTYPE_TO_IDX = {n: i for i, n in enumerate(_RESTYPE_ORDER)}
DESIGNED_RESTYPE_IDX = _RESTYPE_TO_IDX["GAP"]
RESTYPE_DIM = 32
assert DESIGNED_RESTYPE_IDX == 31 and RESTYPE_DIM == len(_RESTYPE_ORDER)

PROTEIN_RES = set(_RESTYPE_ORDER[:20])
RNA_RES = {"A", "C", "G", "U"}
DNA_RES = {"DA", "DC", "DG", "DT"}
PURINE_RES = {"DA", "DG", "A", "G"}
PYRIMIDINE_RES = {"DC", "DT", "C", "U"}
_ELEMENT_TO_ATOMIC_NUMBER = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "CL": 17, "BR": 35, "I": 53,
    "ZN": 30,  # p20: 1j79's real active-site Zn2+ ions -- the first metal
               # ion ligand this port has grounded against a real capture.
}


# -- encoders ---------------------------------------------------------------
def _encode_atom_names_like_af3(names: Sequence[str]) -> np.ndarray:
    """AF3 atom-name encoding: each name padded to 4 chars, each char one-hot
    over 64 bins where bin = ord(c) - 32 (printable ASCII starting at space).
    Returns [N, 4, 64] float32. Matches atomworks._encode_atom_names_like_af3."""
    out = np.zeros((len(names), 4, 64), dtype=np.float32)
    for i, name in enumerate(names):
        s = (name or "")[:4].ljust(4)
        for j, ch in enumerate(s):
            b = ord(ch) - 32
            if 0 <= b < 64:
                out[i, j, b] = 1.0
    return out


def _restype_onehot(res_names: Sequence[str]) -> np.ndarray:
    """[I, 32] one-hot restype (designed slot at DESIGNED_RESTYPE_IDX=31)."""
    out = np.zeros((len(res_names), RESTYPE_DIM), dtype=np.float32)
    for i, r in enumerate(res_names):
        idx = _RESTYPE_TO_IDX.get(str(r).strip().upper(), DESIGNED_RESTYPE_IDX)
        out[i, idx] = 1.0
    return out


# -- structure loading (biotite; atomworks-free) -----------------------------
def load_structure(path: str | Path):
    """Parse a PDB/CIF into a biotite AtomArray with the annotations the
    featurizer needs. This replaces atomworks.io.parser.parse for the
    protein/NA case; it does NOT reproduce atomworks' bond perception,
    assembly building, or CCD normalisation (parity-ungated).

    Heavy atoms only (matches the reference's universal heavy-atom-only
    convention): the protein path already gets this implicitly (hydrogens
    aren't in the "dense" atom14 scheme so they're silently skipped), but an
    NA input file that models explicit hydrogens (e.g. an NMR structure)
    needs an explicit drop here — verified vs a real reference capture
    (1q75.pdb/RNA_basic, which is H-explicit; without this filter L came out
    550 instead of the reference's 386)."""
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io.pdbx import CIFFile, get_structure

    p = Path(path)
    if p.suffix.lower() in (".cif", ".mmcif"):
        cf = CIFFile.read(str(p))
        arr = get_structure(cf, model=1)
    else:
        pf = PDBFile.read(str(p))
        arr = pf.get_structure(model=1)
    arr = arr[arr.element != ""] if hasattr(arr, "element") else arr
    arr = arr[arr.element != "H"] if hasattr(arr, "element") else arr
    return arr


@dataclass
class _Residue:
    chain: str
    res_id: int
    res_name: str
    atom_names: list[str]
    coord: np.ndarray  # [n_atoms, 3]
    elements: list[str]


def _group_residues(arr) -> list[_Residue]:
    """Group a biotite AtomArray into per-residue records (one token per
    residue for protein/NA; ligands are handled separately)."""
    import biotite.structure as struc
    starts = list(struc.get_residue_starts(arr))
    stops = list(struc.get_residue_starts(arr, add_exclusive_stop=True))
    res = []
    for k, s in enumerate(starts):
        e = stops[k + 1] if k + 1 < len(stops) else len(arr)
        sub = arr[s:e]
        res.append(_Residue(
            chain=str(sub.chain_id[0]), res_id=int(sub.res_id[0]),
            res_name=str(sub.res_name[0]).strip().upper(),
            atom_names=[str(a).strip() for a in sub.atom_name],
            coord=np.asarray(sub.coord, dtype=np.float32),
            elements=[str(e).strip().upper() for e in sub.element],
        ))
    return res


def _is_protein(r: _Residue) -> bool:
    return r.res_name in PROTEIN_RES


def _is_na(r: _Residue) -> bool:
    return r.res_name in DNA_RES or r.res_name in RNA_RES


def _central_atom_name(res_name: str) -> str | None:
    """The single "representative atom" name for a DNA/RNA residue (the
    reference's ``is_ca``/``is_central`` flag) — the base ring-center atom:
    C4 for purines, C2 for pyrimidines. Verified vs real captures (dsDNA_basic
    -style and RNA_basic-style)."""
    if res_name in PURINE_RES:
        return "C4"
    if res_name in PYRIMIDINE_RES:
        return "C2"
    return None


def _motif_atom_layout(r: _Residue, keep: frozenset | None = None):
    """Real heavy atoms of a motif (fixed-seq) residue, in the INPUT
    STRUCTURE'S real atom order, renamed to the generic atom14 template via
    the dense-scheme slot lookup — NOT padded, NOT slot-sorted. Missing atoms
    (absent from the input structure) are simply skipped, matching the
    reference (it only ever sees the atoms present in the parsed structure).
    ``keep`` (F4, unindexed tokens only): when not None, real atoms NOT named
    in it are dropped too — this is how the reference's `select_fixed_atoms`
    atom-subsetting reaches an unindexed motif token (see module docstring);
    None (the F1/F6/p14 default) keeps every real atom, unchanged.

    Emission order matters and is NOT the same as slot order: a residue whose
    real PDB atom listing doesn't already happen to match the canonical
    dense-scheme slot order (e.g. a TRP with CE2/CE3 listed before NE1) keeps
    its real order in the reference — only the per-atom NAME is remapped to
    the generic V-slot label. Verified vs a real reference capture (p14,
    ``scripts/rfd3_port/parity_artifacts/parity_unindex.py``, residue A100
    TRP): getting this wrong silently permutes ``motif_pos``/atom-name
    features for any residue with non-canonical side-chain atom ordering in
    its input file (didn't manifest on the p12 F1/F6 fixture by coincidence).
    Returns (names, coord[n,3], is_virtual[n]=False, elements=None) — elements
    is always None for protein: ``ref_element`` is never filled for protein
    atoms regardless (see module docstring), so no per-atom element is needed."""
    scheme = _DENSE_ATOM14_SCHEME.get(r.res_name)
    if scheme is None:
        raise NotImplementedError(f"no dense atom14 scheme for motif residue {r.res_name!r}")
    slot_by_name = {name: slot for slot, name in enumerate(scheme) if name is not None}
    names: list[str] = []
    coord: list[np.ndarray] = []
    seen: set[str] = set()
    for real_name, c in zip(r.atom_names, r.coord):
        if keep is not None and real_name not in keep:
            continue
        slot = slot_by_name.get(real_name)
        if slot is None or real_name in seen:
            continue
        seen.add(real_name)
        names.append(ATOM14_ATOM_NAMES[slot])
        coord.append(c)
    coord_arr = np.asarray(coord, dtype=np.float32) if coord else np.zeros((0, 3), dtype=np.float32)
    return names, coord_arr, np.zeros(len(names), dtype=bool), None


def _na_atom_layout(r: _Residue):
    """Real heavy atoms of a DNA/RNA motif residue, verbatim: real names, real
    order, no scheme lookup, no renaming (``PadTokensWithVirtualAtoms``'s
    ``is_residue`` gate is protein-only, so non-protein tokens never get
    V-slot-relabeled or padded — see module docstring). Elements are the real
    parsed per-atom element (needed for ``ref_element``, which — unlike
    protein — IS filled for NA). Returns (names, coord[n,3], is_virtual[n]
    =False, elements[n])."""
    return list(r.atom_names), r.coord.copy(), np.zeros(len(r.atom_names), dtype=bool), list(r.elements)


def _designed_atom_layout():
    """Full 14-slot template for a designed (sequence-unknown) residue: the
    5 backbone+CB slots are real (undetermined, coord 0); V0..V8 are virtual
    pad atoms (PadTokensWithVirtualAtoms). Returns (names, coord[14,3],
    is_virtual[14], elements=None)."""
    names = list(ATOM14_ATOM_NAMES)
    coord = np.zeros((14, 3), dtype=np.float32)
    is_virtual = np.zeros(14, dtype=bool)
    is_virtual[5:] = True
    return names, coord, is_virtual, None


# -- contig -> token plan ----------------------------------------------------
@dataclass
class _Token:
    chain: str
    res_id: int          # input PDB res_id (motif) or assigned (designed)
    res_name: str        # real name (motif) or "" (designed -> DESIGNED_RESTYPE_IDX)
    is_motif: bool       # BROAD: has known identity (fixed coord OR fixed seq OR
                         # unindexed) -> drives ref_motif_token_type/ref_plddt/
                         # motif_token_class. For protein/NA this always coincides
                         # with "fully fixed coord+seq" (both True together); a
                         # ligand can be is_motif=True (known chemical identity)
                         # while is_fixed_coord=False (diffused position) -- see
                         # is_fixed_coord/is_fixed_seq below.
    is_designed: bool    # diffused
    is_unindexed: bool    # unindexed motif (from the unindex field)
    is_chain_break_before: bool  # /0 precedes this token
    residue: _Residue | None  # source residue for motif tokens, else None
    unindex_new_island: bool = False  # this unindexed token starts a new RPE-leak island
    is_ligand: bool = False  # F3/F4: one atom == one token (atomize)
    is_fixed_coord: bool | None = None  # None => derive from is_motif (protein/NA)
    is_fixed_seq: bool | None = None    # None => derive from is_motif (protein/NA)
    ligand_atom_name: str | None = None  # real atom name (ligand tokens only)
    kept_atom_names: frozenset | None = None  # F4: unindexed + select_fixed_atoms
                         # atom subset (None => keep every real atom, the F6/p14 default)

    @property
    def fixed_coord(self) -> bool:
        return self.is_motif if self.is_fixed_coord is None else self.is_fixed_coord

    @property
    def fixed_seq(self) -> bool:
        return self.is_motif if self.is_fixed_seq is None else self.is_fixed_seq


def _unindexed_kept_atom_names(spec: InputSpecification, chain: str, res_id: int,
                                real_names: Sequence[str]) -> frozenset | None:
    """F4: which of an unindexed residue's real atoms actually enter the
    design, per `select_fixed_atoms` (real reference:
    `input_parsing.py::_build_init` does
    `unindexed_tokens[k] = tok[tok.is_motif_atom_with_fixed_coord]` BEFORE any
    tokenization — a dict-keyed selection SUBSETS the token's real atoms, it
    does not just flag them; see module docstring). Returns None (keep every
    real atom — the F6/p14-verified default) when `select_fixed_atoms` is the
    global True/False default or a dict that doesn't name this residue's
    `{chain}{res_id}` key (verified vs `apply_selections`: an un-named
    residue keeps the annotation's init default, which is True/keep-all)."""
    sel = spec.select_fixed_atoms
    if sel is None or isinstance(sel, bool):
        return None if (sel is None or sel) else frozenset()
    if isinstance(sel, dict):
        key = f"{chain}{res_id}".upper()
        for k, v in sel.items():
            if str(k).strip().upper() == key:
                mask = _atom_selection_mask(_parse_atom_spec(v), real_names)
                return frozenset(np.asarray(real_names)[mask].tolist())
        return None
    raise NotImplementedError(
        f"select_fixed_atoms={sel!r} (a contig-string selection) on an "
        "unindexed protein residue is not supported this pass — p17+"
    )


def _plan_unindexed_tokens(spec: InputSpecification, residues: list[_Residue]) -> list[_Token]:
    """Unindexed motif tokens (``spec.unindex``), appended at the END of the
    token list — matches the reference (``accumulate_components`` places
    unindexed components after the main contig; ``UnindexFlaggedTokens``
    reorders/expands them, but at inference they are already physically last).

    Scoped this pass (p14, grounded via a real local reference capture —
    ``scripts/rfd3_port/parity_artifacts/parity_unindex.py``): plain contig
    components only (a single indexed residue, or an indexed ``-`` RANGE which
    ties the residues together / "leaks" their relative order to the model).
    The doc-described numeric-offset-tie syntax (``A11,0,A12`` / ``A11,3,A12``)
    and dict-form per-atom fixing are NOT implemented — both are genuinely
    ambiguous from the reference source alone (the offset digit does not
    obviously survive into ``get_motif_components_and_breaks``'s breaks array
    the way the docs describe) and were not capture-verified this pass; they
    raise NotImplementedError with a pointer here rather than guess.

    A residue is "tied" (leaked) to the PRECEDING residue in the same ``-``
    range (RPE may reveal their relative sequence position); it is masked
    (never leaked) from every other token — indexed motif, designed, AND any
    other unindexed island — per the captured ``unindexing_pair_mask`` (see
    ``UnindexFlaggedTokens.create_unindexed_masks``: group id = cumsum of
    per-token "new island" flags; same group -> leak allowed, else masked).
    """
    if spec.unindex is None:
        return []
    if not isinstance(spec.unindex, str):
        raise NotImplementedError("dict-form unindex (per-atom fixing) — p14+")
    by_key = {(r.chain, r.res_id): r for r in residues}
    comps = spec.parsed_unindex()
    tokens: list[_Token] = []
    for c in comps:
        if not isinstance(c, Indexed):
            raise NotImplementedError(
                f"unindex component {c!r} not supported this pass (p14): only "
                "plain indexed residues/ranges ('A244' or 'A11-12'); the "
                "numeric-offset-tie syntax and '/0' inside unindex are out of scope"
            )
        for k, rid in enumerate(range(c.start, c.end + 1)):
            r = by_key.get((c.chain, rid))
            if r is None:
                raise ValueError(f"unindex references {c.chain}{rid} not present in input structure")
            if not _is_protein(r):
                raise NotImplementedError("non-protein unindexed motif (NA/ligand) — p15+")
            kept = _unindexed_kept_atom_names(spec, c.chain, rid, r.atom_names)
            tokens.append(_Token(c.chain, rid, r.res_name, True, False, True,
                                 False, r, unindex_new_island=(k == 0),
                                 kept_atom_names=kept))
    return tokens


def _fresh_chain_letter(used: set[str]) -> str:
    """Allocate a synthetic chain letter for a Designed/DesignedRange segment
    that does not reuse any real input chain or previously-assigned synthetic
    chain. Verified vs a real reference capture (dsDNA_basic-style): a
    designed segment immediately after a contig chain break (``/0``) gets its
    own new chain/entity, NOT the preceding indexed block's chain letter."""
    import string
    for letter in string.ascii_uppercase:
        if letter not in used:
            used.add(letter)
            return letter
    raise NotImplementedError("more than 26 chains — p15+")


# -- symmetry (F5) ------------------------------------------------------------
# Bit-exact port of rfd3.inference.symmetry.frames.get_cyclic_frames /
# get_dihedral_frames (verified against a real local capture — see module
# docstring's F5 grounding section). Each frame is a rigid (R[3,3], t[3])
# transform; t is always the zero vector for both symmetry kinds (only
# rotation about the origin, no translation).
_SYMMETRY_ID_RE = re.compile(r"^([CD])(\d+)$")


def _cyclic_frames(order: int) -> list[tuple[np.ndarray, np.ndarray]]:
    frames = []
    for i in range(order):
        angle = 2 * np.pi * i / order
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]], dtype=np.float32)
        frames.append((R, np.zeros(3, dtype=np.float32)))
    return frames


def _dihedral_frames(order: int) -> list[tuple[np.ndarray, np.ndarray]]:
    frames = []
    for i in range(order):
        angle = 2 * np.pi * i / order
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]], dtype=np.float32)
        phi = angle + np.pi / order
        u = np.array([np.cos(phi), np.sin(phi), 0], dtype=np.float32)
        flip = -np.eye(3, dtype=np.float32) + 2 * np.outer(u, u)
        frames.append((R, np.zeros(3, dtype=np.float32)))
        frames.append((R @ flip, np.zeros(3, dtype=np.float32)))
    return frames


def _symmetry_frames(sym_conf: Mapping) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parse `spec.symmetry` (e.g. ``{"id": "C3"}``) into a list of rigid
    transforms, one per symmetric copy (transform 0 == the ASU/identity).
    Only cyclic (``C<n>``) and dihedral (``D<n>``) groups are supported —
    matches the real reference's own stated scope ("only C and D symmetry
    types are supported currently", `docs/examples/symmetry.md`)."""
    sym_id = str(sym_conf.get("id", "")).strip().upper()
    m = _SYMMETRY_ID_RE.match(sym_id)
    if not m:
        raise NotImplementedError(f"symmetry id {sym_id!r} not supported (only C<n>/D<n>) — p18+")
    kind, order = m.group(1), int(m.group(2))
    return _cyclic_frames(order) if kind == "C" else _dihedral_frames(order)


def _symmetrize_tokens(asu_tokens: list["_Token"], frames: list[tuple[np.ndarray, np.ndarray]],
                        used_chains: set[str], replica_chains: list[str] | None = None,
                        ) -> tuple[list["_Token"], list[int], set[str], list[str]]:
    """Replicate an ASU token list (or ASU-token SUBSET, p19: e.g. just an
    unindexed-motif block, see module docstring's "unindexed-motif
    replication" grounding) into ``len(frames)`` symmetric copies. Copy 0 is
    the ASU itself, UNCHANGED (transform_id=0, its own real chain letter,
    never reassigned — matches the reference's own transform_id=0 being a
    no-op copy); copies 1..N-1 each get a fresh chain letter (own
    `residue_index` run, verified vs a real capture). ``replica_chains``
    (p19): pass a PRECOMPUTED list (from an earlier call) to make a SECOND,
    disjoint ASU-token group (e.g. an unindexed-motif block) share the exact
    same per-replica chain letters as the first (verified vs a real capture:
    a replica's unindexed-motif copy shares `asym_id`/`entity_id` with its
    subunit's own main chain) — ``None`` (p18 default) allocates fresh ones.
    Returns (all tokens in ASU-then-replica order, each token's transform id,
    the set of chains belonging to this one symmetric group — needed by the
    entity_id/sym_id special case in `featurize()`, and the replica_chains
    list itself, for reuse by a second call).

    A replica token that carries a REAL residue (a real motif atom, e.g. an
    unindexed catalytic residue "within a subunit") must have that residue's
    REAL COORDINATES rotated by the replica's own frame, not just its
    `.chain` relabeled — a purely-designed token (`residue=None`) has no
    coordinate to transform either way, which is why this was invisible in
    p18's fully-unconditional-only scope (every replicated token was
    `residue=None`) and only surfaced as a real bug this pass, once a real
    motif was replicated for the first time (verified vs a real capture: the
    bug produced a near-zero `motif_pos` — both replicas' underlying
    `_Residue` objects were literally the SAME shared, untransformed real
    coordinate, so centering on their mean collapsed both to ~0)."""
    if replica_chains is None:
        replica_chains = [_fresh_chain_letter(used_chains) for _ in range(1, len(frames))]
    assert len(replica_chains) == len(frames) - 1
    tokens = list(asu_tokens)
    transform_ids = [0] * len(asu_tokens)
    sym_chains = {tk.chain for tk in asu_tokens} | set(replica_chains)
    for tid, chain in enumerate(replica_chains, start=1):
        R, t = frames[tid]
        for tk in asu_tokens:
            if tk.residue is not None:
                new_residue = _dc_replace(tk.residue, coord=tk.residue.coord @ R.T + t)
                tokens.append(_dc_replace(tk, chain=chain, residue=new_residue))
            else:
                tokens.append(_dc_replace(tk, chain=chain))
            transform_ids.append(tid)
    return tokens, transform_ids, sym_chains, replica_chains


def _kabsch_align(X_fixed: np.ndarray, X_moving: np.ndarray) -> np.ndarray:
    """Bit-faithful numpy port of ``rfd3.inference.symmetry.frames._align``
    (the numpy code path — the real reference itself calls this with plain
    numpy `biotite` coordinates, never torch, for
    `get_symmetry_frames_from_atom_array`), TRANSLATIONS DISCARDED (only the
    rotation `R` is used by the caller, matching the reference discarding
    both returned means too — see module docstring's "get_symmetry_frames_
    from_atom_array" grounding for why a pure rotation is sufficient once the
    whole input has been pre-centered via `_center_symmetric_src_residues`).
    Returns `R` such that ``R @ (X_moving - mean(X_moving)) ~= X_fixed -
    mean(X_fixed)``."""
    u_fixed = X_fixed.mean(axis=0)
    u_moving = X_moving.mean(axis=0)
    Xf = X_fixed - u_fixed
    Xm = X_moving - u_moving
    C = Xf.T @ Xm
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    R = U @ Vt
    F = np.eye(3, dtype=R.dtype)
    F[-1, -1] = np.sign(np.linalg.det(R))
    R = U @ F @ Vt
    return R.astype(np.float32)


def _center_symmetric_src_residues(all_residues: list[_Residue]) -> list[_Residue]:
    """Bit-faithful port of ``rfd3.inference.symmetry.symmetry_utils.
    center_symmetric_src_atom_array``: shift EVERY real residue's real
    coordinates (protein AND any motif/ligand/NA) by the negative mean of
    PROTEIN-only real atoms, once, before any token planning — the real
    reference applies this unconditionally whenever `symmetry` is set on a
    spec with a real `input` structure (regardless of whether frames end up
    Kabsch-derived or closed-form). See module docstring's F5+motif
    grounding for why this is required for `get_symmetry_frames_from_
    atom_array`'s discarded-translation frames to correctly reproduce a real
    subunit's true position."""
    protein_coords = [r.coord for r in all_residues if _is_protein(r)]
    if not protein_coords:
        return all_residues
    com = np.concatenate(protein_coords, axis=0).mean(axis=0)
    return [_dc_replace(r, coord=r.coord - com) for r in all_residues]


def _symmetry_frames_from_structure(all_residues: list[_Residue], n_frames: int,
                                     ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bit-faithful (scope-narrowed) port of ``rfd3.inference.symmetry.
    frames.get_symmetry_frames_from_atom_array``: derive `n_frames` rigid
    (R, t=0) transforms from the RAW input structure's own real protein
    chains (Kabsch alignment), rather than the closed-form origin-rotation
    formula (see module docstring's grounding). Scoped this pass to a single
    real symmetric protein entity whose chain count equals `n_frames` exactly
    (every real reference example this port grounds against is this shape) —
    a genuinely heteromeric or mismatched-multiplicity symmetric input raises
    NotImplementedError rather than guess at the real function's general
    multi-entity multiplicity-detection logic."""
    protein = [r for r in all_residues if _is_protein(r)]
    by_chain: dict[str, list[_Residue]] = {}
    for r in protein:
        by_chain.setdefault(r.chain, []).append(r)
    seq_by_chain = {c: tuple(r.res_name for r in rs) for c, rs in by_chain.items()}
    groups: dict[tuple, list[str]] = {}
    for c, seq in seq_by_chain.items():
        groups.setdefault(seq, []).append(c)
    candidates = [chains for chains in groups.values() if len(chains) == n_frames]
    if len(candidates) != 1:
        raise NotImplementedError(
            "get_symmetry_frames_from_atom_array: expected exactly one real "
            f"protein entity with {n_frames} identical-sequence chains (the "
            f"symmetry id's own order), found candidate group sizes "
            f"{sorted(len(v) for v in groups.values())} — a genuinely "
            "heteromeric or mismatched-multiplicity symmetric input is not "
            "supported this pass (p19+)"
        )
    chains_to_consider = sorted(candidates[0])
    ref_chain = chains_to_consider[0]

    def _flat_coord(c: str) -> np.ndarray:
        return np.concatenate([r.coord for r in by_chain[c]], axis=0)

    ref_coord = _flat_coord(ref_chain)
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for c in chains_to_consider:
        coord = _flat_coord(c)
        if coord.shape != ref_coord.shape:
            raise NotImplementedError(
                f"get_symmetry_frames_from_atom_array: chain {c!r} has "
                f"{coord.shape[0]} real protein atoms, reference chain "
                f"{ref_chain!r} has {ref_coord.shape[0]} — unequal subunit "
                "sizes not supported this pass (p19+)"
            )
        R = _kabsch_align(coord, ref_coord)
        frames.append((R, np.zeros(3, dtype=np.float32)))
    return frames


def _parse_is_unsym_motif(sym_conf: Mapping) -> set[str] | None:
    """`symmetry.is_unsym_motif`: comma-separated list of ligand CCD codes
    and/or contig-style chain+residue(-range) names that should NOT be
    symmetrized (``rfd3.inference.symmetry.contigs.
    expand_contig_unsym_motif``/``get_unsym_motif_mask``, verified against a
    real capture of the real `unsym_C3_6t8h` example). A residue RANGE
    (``"Y1-11"``) expands to individual ``{chain}{res_id}`` names; a bare
    token with no ``-`` is kept as-is (a ligand CCD code like ``"HEM"``, or a
    single residue like ``"M52"``)."""
    raw = sym_conf.get("is_unsym_motif") if isinstance(sym_conf, Mapping) else None
    if not raw:
        return None
    names: list[str] = []
    for n in str(raw).split(","):
        n = n.strip()
        if not n:
            continue
        m = re.match(r"^([A-Za-z]+)(\d+)-(\d+)$", n)
        if m:
            chain, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
            names.extend(f"{chain}{i}" for i in range(lo, hi + 1))
        else:
            names.append(n)
    return set(names)


def _token_matches_unsym(tk: "_Token", names: set[str]) -> bool:
    if tk.is_ligand:
        return tk.res_name in names
    return f"{tk.chain}{tk.res_id}" in names


def _plan_tokens_from_contig(spec: InputSpecification, residues: list[_Residue]) -> tuple[list[_Token], set]:
    """Map a parsed contig + the parsed structure's residues into an ordered
    token plan. Indexed components pull residues from the input structure
    (motif, fixed coord+seq, protein OR DNA/RNA); Designed/DesignedRange
    components become diffused (protein) tokens; ChainBreak marks the next
    token. Unindexed motif tokens are NOT appended here — see `featurize()`,
    which places them after any ligand tokens (F4: verified vs a real
    reference capture of `enzyme_design.md`'s "NAI,ACT" + unindexed-catalytic-
    residue example that ligand tokens physically precede unindexed ones,
    correcting this port's pre-p17 assumption that unindexed was always
    physically last — true only when no ligand coexists, per p11-p14/p16's
    own tests, which never combined the two)."""
    by_key = {(r.chain, r.res_id): r for r in residues}
    comps = spec.parsed_contig()
    tokens: list[_Token] = []
    break_before_next = False
    indexed_keys: set[tuple[str, int]] = set()
    # NOT pre-seeded with every real residue's chain (F4 finding, see below):
    # only chains actually CONSUMED so far by the output token list (indexed
    # components as they're visited, plus previously-allocated designed
    # chains) are avoided -- a real chain the contig never touches is fair
    # game for a designed segment to reuse. Verified vs a real reference
    # capture (enzyme_design.md, a bare "length"-only spec with unindex-only
    # catalytic residues on real chain "A"): the reference's sole designed
    # segment IS named "A" (`accumulate_components(..., start_chain="A")`)
    # and the later-appended unindexed catalytic residues (also real chain
    # "A") end up sharing that SAME asym_id/entity/residue_index run, not a
    # fresh one -- pre-seeding `used_chains` from every real residue (this
    # port's pre-p17 behavior) wrongly steered the sole designed segment away
    # from "A" whenever ANY part of the input structure used that letter,
    # even a part never referenced by `contig` at all.
    used_chains: set[str] = set()
    for c in comps:
        if isinstance(c, ChainBreak):
            break_before_next = True
            continue
        if isinstance(c, Indexed):
            used_chains.add(c.chain)
            for rid in range(c.start, c.end + 1):
                r = by_key.get((c.chain, rid))
                if r is None:
                    raise ValueError(f"contig indexes {c.chain}{rid} not present in input structure")
                if not (_is_protein(r) or _is_na(r)):
                    raise NotImplementedError("ligand/enzyme indexed motif (F3/F4) — p15+")
                tokens.append(_Token(c.chain, rid, r.res_name, True, False, False,
                                     break_before_next, r))
                indexed_keys.add((c.chain, rid))
                break_before_next = False
            continue
        if isinstance(c, (Designed, DesignedRange)):
            n = c.length if isinstance(c, Designed) else (c.lo + c.hi) // 2
            # A designed segment continues the preceding block's chain UNLESS
            # a chain break (or nothing yet) precedes it, in which case it
            # gets a brand-new synthetic chain (verified vs a real capture).
            if break_before_next or not tokens:
                chain = _fresh_chain_letter(used_chains)
            else:
                chain = tokens[-1].chain
            base = (tokens[-1].res_id + 1) if tokens and tokens[-1].chain == chain else 1
            for k in range(n):
                tokens.append(_Token(chain, base + k, "", False, True, False,
                                     break_before_next, None))
                break_before_next = False
            continue
        raise NotImplementedError(f"contig component {c!r} not supported this pass (p12)")
    return tokens, indexed_keys


def _token_kind(tk: "_Token") -> str:
    """'protein' | 'dna' | 'rna' | 'ligand' for a token (designed tokens,
    res_name=="", are always protein in this port's scope — NA is never
    diffused/designed in the documented use cases, only ever a fixed binder
    target)."""
    if tk.is_ligand:
        return "ligand"
    if not tk.res_name or tk.res_name in PROTEIN_RES:
        return "protein"
    if tk.res_name in DNA_RES:
        return "dna"
    if tk.res_name in RNA_RES:
        return "rna"
    raise NotImplementedError(f"unrecognized residue {tk.res_name!r} — p17+")


# -- ligand (F3/F4) -----------------------------------------------------------
# Real per-atom CCD template (name -> element/charge/reference-conformer coord)
# + intra-ligand bond graph, reused from tt_bio's EXISTING Boltz-2/Protenix-v2
# bundled CCD rdkit-mol library (tt_bio.data.mol.load_molecules, default
# ~/.boltz/mols) rather than re-vendoring RDKit/CCD conformer generation.
#
# `ref_pos` is reference-CONFORMER geometry, not ground-truth identity: the
# real reference (rfd3.transforms.design_transforms.CreateDesignReferenceFeatures
# -> atomworks.ml.transforms.af3_reference_molecule.get_af3_reference_molecule_features)
# generates it via a STOCHASTIC RDKit ETKDG embed (a fresh random seed/rotation
# per run, verified: `ccd_code_to_rdkit_with_conformers`/`random_rigid_augmentation`
# both draw an unseeded random value absent an explicit seed) — so no single
# reference capture's `ref_pos` is "the" bit-exact target. This port's own
# Protenix-v2 host featurizer already documents and relies on exactly this
# invariance (tt_bio/protenix_data.py:466, "the reference uses a STOCHASTIC
# RDKit conformer, so any valid one folds correctly"); the same principle is
# applied here rather than re-derived.
def _ligand_template(ccd_code: str, mol_dir: str | None = None) -> dict:
    from .data.mol import load_molecules
    mol_dir = mol_dir or _os.path.expanduser("~/.boltz/mols")
    mols = load_molecules(mol_dir, [ccd_code])
    mol = mols[ccd_code]
    conf = mol.GetConformer(0)
    names: list[str] = []
    elements: list[str] = []
    charges: list[int] = []
    coords: list[tuple[float, float, float]] = []
    idx_by_rdkit_idx: dict[int, int] = {}
    for a in mol.GetAtoms():
        if a.GetAtomicNum() <= 1:  # heavy atoms only, matches this port's convention
            continue
        nm = a.GetProp("name") if a.HasProp("name") else a.GetSymbol().upper()
        p = conf.GetAtomPosition(a.GetIdx())
        idx_by_rdkit_idx[a.GetIdx()] = len(names)
        names.append(nm)
        elements.append(a.GetSymbol().upper())
        charges.append(int(a.GetFormalCharge()))
        coords.append((p.x, p.y, p.z))
    bonds: list[tuple[int, int]] = []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in idx_by_rdkit_idx and v in idx_by_rdkit_idx:
            bonds.append((idx_by_rdkit_idx[u], idx_by_rdkit_idx[v]))
    return {
        "names": names, "elements": elements, "charges": charges,
        "coord": np.asarray(coords, dtype=np.float32), "bonds": bonds,
    }


def _resolve_ligand_atom_selection(sel_value, code: str, not_selected: "AtomSelection" = None) -> AtomSelection:
    """Resolve a select_fixed_atoms/select_buried/select_exposed field value to
    an ``AtomSelection`` for a ligand, keyed by CCD code (e.g. ``{"IAI": "C1,C2"}``)
    — a SEPARATE key convention from protein/NA's ``{chain}{res_id}`` (verified
    vs a real reference capture + rfd3.inference.parsing.canonicalize_, which
    resolves a bare ligand-code dict key via `unravel_components` to the
    ligand's actual chain+res_id before the same per-atom-name lookup protein/
    NA use).

    A residue with NO entry in the dict (or the field not provided at all)
    is genuinely SKIPPED by the reference's per-residue `apply_selections`
    (`input_parsing.py`: `atom_names_sele = selection.get(...); if
    atom_names_sele is None: continue`) — it keeps whichever GLOBAL init
    value that annotation started at, UNTOUCHED. ``not_selected`` is that
    per-FIELD init value, as an ``AtomSelection`` (``none_()`` default,
    correct for select_buried/select_exposed's `rasa_bin`, whose "no
    override" IS "no atoms"). ``select_fixed_atoms`` is DIFFERENT:
    ``is_motif_atom_with_fixed_coord``'s global init
    (`REQUIRED_CONDITIONING_ANNOTATION_VALUES`) is ``True`` — a ligand code
    absent from `select_fixed_atoms` (p20 fix; the pre-p20 `none_()` default
    here was only coincidentally untested: every fixture through p19 either
    omitted `select_fixed_atoms` entirely for a spec with no ligand, or
    explicitly listed EVERY ligand code including as `""` — verified vs a
    real reference capture of `unindexed_C2_1j79`'s ORO/Zn, absent from that
    spec's `select_fixed_atoms`, which are ALL fixed in the real capture,
    not diffused) — its caller passes ``not_selected=AtomSelection.all_()``."""
    if not_selected is None:
        not_selected = AtomSelection.none_()
    if sel_value is None:
        return not_selected
    if isinstance(sel_value, bool):
        return AtomSelection.all_() if sel_value else AtomSelection.none_()
    if isinstance(sel_value, dict):
        for k, v in sel_value.items():
            if str(k).strip().upper() == code:
                return _parse_atom_spec(v)
        return not_selected
    raise NotImplementedError(
        f"select_* value {sel_value!r} not supported for ligand atoms "
        "(a contig-string selection targeting a ligand) — p17+"
    )


def _atom_selection_mask(sel: AtomSelection, names: Sequence[str]) -> np.ndarray:
    if sel.shorthand == "ALL":
        return np.ones(len(names), dtype=bool)
    if sel.shorthand in ("TIP", "BKBN"):
        raise NotImplementedError(f"{sel.shorthand} shorthand for ligand atoms — p17+")
    if not sel.atoms:
        return np.zeros(len(names), dtype=bool)
    return np.isin(np.asarray(names), np.asarray(sorted(sel.atoms)))


def _ligand_codes(spec: InputSpecification) -> list[str]:
    """F4: `ligand` can name multiple DIFFERENT CCD codes, comma-separated
    (real ``enzyme_design.md`` example: ``"ligand": "NAI,ACT"`` — a cofactor
    plus a separate small-molecule product, each a single instance)."""
    return [c.strip().upper() for c in spec.ligand.split(",") if c.strip()]


def _plan_ligand_tokens(spec: InputSpecification, all_residues: list[_Residue],
                         used_chains: set[str]) -> list[_Token]:
    """One token PER ATOM (AF3/RFD3 "atomize" convention — verified vs a real
    reference capture: PadTokensWithVirtualAtoms's `is_residue` gate excludes
    every non-protein-non-unindexed token, so a ligand is never grouped into a
    per-residue token like protein/NA are). Real coordinates/atom identity come
    from the actual input-structure ligand residue (matched by CCD code).

    Every matching residue (regardless of code) is grouped by its real
    SOURCE chain (``residue.chain``, i.e. which real input chain it was
    physically deposited on) — each source-chain group gets its OWN fresh
    synthetic output chain (p20, generalizing the p17 F4 rule). A
    non-symmetric input's ligand instances typically all share one real
    source chain, so this reduces to exactly the pre-existing "multiple
    DIFFERENT codes share ONE fresh chain" F4 behavior verified against
    ``enzyme_design.md``'s ``"NAI,ACT"`` (both land on the SAME `asym_id`).
    A symmetric input's ligand is typically physically duplicated once per
    real subunit (verified vs ``unindexed_C2_1j79``'s real ORO+2×Zn per
    chain) — those duplicates land on DIFFERENT source chains and so get
    split into one fresh chain per subunit (see module docstring's
    "ligand + symmetry" grounding). Multiple instances of the SAME code
    (p20; e.g. 1j79's two ZN ions per subunit) are supported: matches are
    taken in each code's own structure-file order — the reference's actual
    cross-instance order is PYTHONHASHSEED-dependent and not a real rule to
    reproduce (see module docstring); this port picks its own deterministic
    order instead of guessing at an accident.

    Each atom's ``is_fixed_coord`` (F4 fix) is resolved HERE, at token-
    construction time, from ``select_fixed_atoms`` — not deferred to the main
    per-atom loop. p16's single-ligand tests never exercised a ligand with
    ANY fixed atom (the ``sm_binder_design.md`` "buried" example fixes none),
    so a real bug went uncaught: leaving ``_Token.is_fixed_coord`` at its
    default (None -> derives from ``is_motif``, always True for a ligand)
    made the design's center-of-mass computation (which runs BEFORE the main
    loop, gated on ``tk.fixed_coord``) silently treat EVERY ligand atom as
    fixed, not just the ``select_fixed_atoms``-selected ones — verified vs a
    real reference capture (enzyme_design.md's ACT ligand fixes only its
    OXT atom; the old code's COM was off by a constant vector, visible as a
    constant-offset ``motif_pos`` mismatch on every fixed-coord atom)."""
    codes = _ligand_codes(spec)
    if len(codes) != len(set(codes)):
        raise NotImplementedError(
            f"ligand code repeated in the `ligand` string itself {spec.ligand!r} "
            "— p17+ (a degenerate spec; name each code once, multiple physical "
            "instances of one code are resolved from the structure, not the string)"
        )
    matches_by_code: dict[str, list[_Residue]] = {}
    for code in codes:
        matches = [r for r in all_residues if r.res_name == code]
        if not matches:
            raise ValueError(f"ligand {code!r} not found in input structure")
        matches_by_code[code] = matches

    source_chains = sorted({r.chain for matches in matches_by_code.values() for r in matches})
    fresh_chain_by_source: dict[str, str] = {}
    for src_chain in source_chains:
        fresh_chain_by_source[src_chain] = _fresh_chain_letter(used_chains)

    tokens: list[_Token] = []
    for code in codes:
        for residue in matches_by_code[code]:
            chain = fresh_chain_by_source[residue.chain]
            fixed_mask = _atom_selection_mask(
                _resolve_ligand_atom_selection(
                    spec.select_fixed_atoms, code, not_selected=AtomSelection.all_()),
                residue.atom_names)
            tokens.extend(
                _Token(chain, residue.res_id, code, True, False, False, False, residue,
                       is_ligand=True, ligand_atom_name=nm, is_fixed_coord=bool(fixed_mask[k]))
                for k, nm in enumerate(residue.atom_names)
            )
    return tokens


def _ligand_atom_layout(tk: "_Token"):
    """Single real heavy atom, verbatim (no renaming/padding — see module
    docstring for the reference's atomize gate). Returns (names, coord[1,3],
    is_virtual[1]=False, elements[1])."""
    idx = tk.residue.atom_names.index(tk.ligand_atom_name)
    return ([tk.ligand_atom_name], tk.residue.coord[idx:idx + 1].copy(),
             np.zeros(1, dtype=bool), [tk.residue.elements[idx]])


def featurize(structure_path: str | Path | None, spec: InputSpecification) -> dict[str, torch.Tensor]:
    """Build the ``f`` feature dict for one design spec from a real PDB/CIF.

    Protein-binder (F1) + motif scaffolding (F6) + nucleic-acid-binder design
    (F2/F8: a fixed-sequence DNA/RNA target + a designed protein binder) +
    cyclic/dihedral symmetric oligomer design (F5, unconditional-only this
    pass), with atom-level parity vs the real reference (see module
    docstring).

    ``structure_path=None`` (F5): a fully-unconditional design (no real input
    structure at all) — matches the real reference, which only requires an
    atom array once `contig` is actually set (even to an all-Designed
    string); a bare `length` field never needs one.
    """
    if structure_path is None:
        all_residues: list[_Residue] = []
    else:
        arr = load_structure(structure_path)
        all_residues = _group_residues(arr)
        if spec.symmetry:
            # Bit-faithful port of center_symmetric_src_atom_array: shift
            # EVERY real residue by -mean(protein-only real atoms), once,
            # before any token planning (see module docstring's F5+motif
            # grounding + `_center_symmetric_src_residues`'s docstring).
            all_residues = _center_symmetric_src_residues(all_residues)
    # Non-polymer residues NOT named by `spec.ligand` (solvent, ions, an
    # unreferenced HETATM) are simply invisible to this featurizer, exactly
    # like a real PDB's crystallographic waters are to a contig that never
    # references them. A contig that DOES try to index one fails with a
    # ValueError ("not present in input structure") since it's filtered out
    # here; the ligand (if any) is planned separately below (it is never
    # indexed via `contig`, only via the `ligand` spec field).
    residues = [r for r in all_residues if _is_protein(r) or _is_na(r)]
    # A pure `length`-only spec (no `contig` at all — e.g. a small-molecule
    # binder design with nothing but a fresh designed chain + a `ligand`) is
    # equivalent to a bare designed-length contig string (`parse_contig`
    # already parses a bare "180-180"/"180" as Designed/DesignedRange).
    contig_spec = spec
    if spec.contig is None and spec.length is not None:
        contig_spec = copy.copy(spec)
        contig_spec.contig = str(spec.length)
    tokens, indexed_keys = _plan_tokens_from_contig(contig_spec, residues)
    # Unindexed motif tokens are planned here (early -- needed below for the
    # F5 "does a real motif exist at all" gate) but physically APPENDED only
    # after any ligand tokens (F4: verified vs a real reference capture --
    # see `_plan_tokens_from_contig`'s docstring for why this differs from
    # this port's pre-p17 assumption).
    unindexed = _plan_unindexed_tokens(spec, residues)
    overlap = indexed_keys & {(tk.chain, tk.res_id) for tk in unindexed}
    if overlap:
        raise ValueError(f"contig and unindex must not overlap, got: {overlap}")
    sym_frames = None
    sym_transform_id_by_token: list[int] | None = None
    unindexed_transform_id: list[int] = [-1] * len(unindexed)
    sym_chains: set[str] = set()
    unsym_motif_names = _parse_is_unsym_motif(spec.symmetry) if spec.symmetry else None
    if spec.symmetry and spec.ligand and unsym_motif_names:
        # `is_unsym_motif` naming a `ligand` CCD code directly is a narrower,
        # still-unsupported combination (see module docstring's "ligand +
        # symmetry" grounding): `_plan_ligand_tokens`'s tokens never pass
        # through the `_token_matches_unsym` split at all (they're appended
        # after it runs), so this would otherwise silently no-op instead of
        # raising.
        overlap_codes = set(_ligand_codes(spec)) & set(unsym_motif_names)
        if overlap_codes:
            raise NotImplementedError(
                f"`is_unsym_motif` naming ligand code(s) {sorted(overlap_codes)!r} "
                "directly is not supported this pass (the reference's own "
                "mechanism (c) already treats every ligand as implicitly "
                "unsym — see module docstring) — p20+"
            )
    replica_chains: list[str] | None = None
    if spec.symmetry:
        # F5: pick the frames FIRST (closed-form, unless a real motif exists
        # to derive them from -- see below), then split the built contig
        # token list into "gets replicated" vs "is_unsym_motif, single copy"
        # (mechanism (b)), then replicate (mechanism (a)/(c) -- see module
        # docstring's F5+motif grounding).
        closed_form_frames = _symmetry_frames(spec.symmetry)
        is_symmetric_motif = bool(spec.symmetry.get("is_symmetric_motif", True))
        has_real_motif = any(tk.fixed_coord for tk in tokens + unindexed)
        if structure_path is not None and has_real_motif and is_symmetric_motif:
            sym_frames = _symmetry_frames_from_structure(all_residues, len(closed_form_frames))
        else:
            sym_frames = closed_form_frames

        if unsym_motif_names:
            sym_part = [tk for tk in tokens if not _token_matches_unsym(tk, unsym_motif_names)]
            unsym_part = [tk for tk in tokens if _token_matches_unsym(tk, unsym_motif_names)]
        else:
            sym_part, unsym_part = tokens, []
        used_chains = {tk.chain for tk in tokens} | {tk.chain for tk in unindexed}
        sym_part, sym_transform_id_by_token, sym_chains, replica_chains = _symmetrize_tokens(
            sym_part, sym_frames, used_chains)
        # is_unsym_motif tokens are never part of any transform (F5 "fixed"
        # sentinel, transform_id=-1) and keep their own real chain letter,
        # unchanged; physically placed AFTER the replicated block (verified
        # vs a real capture of unsym_C3_6t8h).
        tokens = sym_part + unsym_part
        sym_transform_id_by_token = sym_transform_id_by_token + [-1] * len(unsym_part)

        if unindexed:
            # F5 + unindexed motif (p19, mechanism (c)/(b) -- see module
            # docstring's F5+motif grounding): same split-then-replicate
            # treatment, reusing the SAME `replica_chains` so a replica's
            # unindexed-motif copy shares asym_id/entity_id with its
            # subunit's own main chain (verified vs a real capture of the
            # unindexed_C2_1j79-minus-ligand variant).
            if unsym_motif_names:
                unind_sym = [tk for tk in unindexed if not _token_matches_unsym(tk, unsym_motif_names)]
                unind_unsym = [tk for tk in unindexed if _token_matches_unsym(tk, unsym_motif_names)]
            else:
                unind_sym, unind_unsym = unindexed, []
            if unind_sym:
                used_chains = {tk.chain for tk in tokens} | set(replica_chains)
                unind_sym, unind_tid, _, _ = _symmetrize_tokens(
                    unind_sym, sym_frames, used_chains, replica_chains=replica_chains)
            else:
                unind_tid = []
            unindexed = unind_sym + unind_unsym
            unindexed_transform_id = unind_tid + [-1] * len(unind_unsym)
    ligand_chains: set[str] = set()
    if spec.ligand:
        # Fresh chain letter(s) must avoid whatever chain the OUTPUT token
        # list has claimed so far (indexed + designed + symmetric-replica
        # tokens) -- NOT every real input chain (see `_plan_tokens_from_contig`'s
        # docstring for the grounded reasoning; a real chain never touched by
        # contig/ligand is fair game and can legitimately coincide with a
        # later unindexed residue's own real chain, verified vs a real
        # reference capture).
        used_chains = {tk.chain for tk in tokens}
        ligand_tokens = _plan_ligand_tokens(spec, all_residues, used_chains)
        ligand_chains = {tk.chain for tk in ligand_tokens}
        tokens = tokens + ligand_tokens
        if spec.symmetry:
            # F5 mechanism (c): a ligand -- symmetric input or not -- is
            # NEVER resymmetrized by the sampler, the same FIXED sentinel as
            # an is_unsym_motif/post-replication-unindexed token (verified
            # vs module docstring's "ligand + symmetry" grounding: every
            # ligand atom in a real capture has sym_transform_id=-1).
            sym_transform_id_by_token = sym_transform_id_by_token + [-1] * len(ligand_tokens)
    tokens = tokens + unindexed
    if spec.symmetry:
        sym_transform_id_by_token = sym_transform_id_by_token + unindexed_transform_id
        # Bit-faithful port of fix_3D_sym_motif_annotations's post-hoc
        # override: EVERY unindexed-motif atom (regardless of which replica
        # transform it came from) is forced back to the F5 "fixed" sentinel
        # -- it was still replicated (real, geometrically-correct per-replica
        # coordinates via mechanism (a)/(c)), but must never be
        # RE-symmetrized by the sampler from an ASU (verified vs a real
        # capture: unindexed_C2_1j79-minus-ligand's replica copy of A250 has
        # sym_transform_id=-1, not its real transform id 1).
        sym_transform_id_by_token = [
            -1 if tk.is_unindexed else tid for tk, tid in zip(tokens, sym_transform_id_by_token)
        ]
    I = len(tokens)
    token_kind = [_token_kind(tk) for tk in tokens]

    # Per-token atom layout (variable count: motif = real heavy atoms only,
    # designed = full 14-slot template, ligand = one real heavy atom). NA/
    # ligand motif tokens keep real atom names/order verbatim (no scheme
    # lookup — see module docstring).
    layouts = [
        _ligand_atom_layout(tk) if kind == "ligand" else
        (_na_atom_layout(tk.residue) if kind in ("dna", "rna") else _motif_atom_layout(tk.residue, tk.kept_atom_names))
        if tk.is_motif else _designed_atom_layout()
        for tk, kind in zip(tokens, token_kind)
    ]
    L = sum(len(nm) for nm, _, _, _ in layouts)

    # The whole design is centered at the center of mass of the real, FIXED-
    # COORD atoms (verified vs a real reference capture: motif_pos ==
    # real_coord - com, where com = mean over every atom whose coordinate is
    # actually held fixed — for protein/NA this coincides with `is_motif`, but
    # a ligand can be `is_motif` (known chemical identity) while NOT
    # fixed-coord (its position is diffused), so it must NOT contribute to
    # centering in that case; gate on `tk.fixed_coord`, not the broader
    # `tk.is_motif`).
    motif_coords = [c for tk, (nm, c, _, _) in zip(tokens, layouts) if tk.fixed_coord and len(nm)]
    if motif_coords:
        com = np.concatenate(motif_coords, axis=0).mean(axis=0)
    else:
        com = np.zeros(3, dtype=np.float32)

    atom_names: list[str] = []
    atom_elements: list[str | None] = []
    atom_coord = np.zeros((L, 3), dtype=np.float32)
    is_virtual = np.zeros(L, dtype=bool)
    is_backbone = np.zeros(L, dtype=bool)
    is_sidechain = np.zeros(L, dtype=bool)
    is_ca = np.zeros(L, dtype=bool)
    is_central = np.zeros(L, dtype=bool)
    is_motif_atom_fixed_coord = np.zeros(L, dtype=bool)
    is_motif_atom_fixed_seq = np.zeros(L, dtype=bool)
    is_motif_atom_unindexed = np.zeros(L, dtype=bool)
    is_na_atom = np.zeros(L, dtype=bool)
    motif_pos = np.zeros((L, 3), dtype=np.float32)
    ref_space_uid = np.zeros(L, dtype=np.int64)
    atom_to_token_map = np.zeros(L, dtype=np.int32)

    token_res_name = []
    token_chain = []
    token_res_id = []
    token_is_motif = np.zeros(I, dtype=bool)
    token_is_unindexed = np.zeros(I, dtype=bool)
    token_break_before = np.zeros(I, dtype=bool)
    token_is_fully_fixed_coord = np.zeros(I, dtype=bool)
    is_ligand_atom = np.zeros(L, dtype=bool)

    # Ligand-only per-atom selections (resolved once per CODE, outside the
    # loop): each ligand instance's own real atom-name order (== emission
    # order, since `_ligand_atom_layout` emits one real atom per token in the
    # SAME order `_plan_ligand_tokens` created them) lets a single boolean
    # mask line up with that instance's atoms. F4: multiple different codes
    # each get their OWN mask/template, keyed by code (a code's atoms are
    # always contiguous in the token list, one block per instance).
    if spec.ligand:
        _lig_buried_mask, _lig_exposed_mask = {}, {}
        _lig_template, _lig_pos_by_name = {}, {}
        for code in _ligand_codes(spec):
            lig_names = [tk.ligand_atom_name for tk in tokens if tk.is_ligand and tk.res_name == code]
            _lig_buried_mask[code] = _atom_selection_mask(
                _resolve_ligand_atom_selection(spec.select_buried, code), lig_names)
            _lig_exposed_mask[code] = _atom_selection_mask(
                _resolve_ligand_atom_selection(spec.select_exposed, code), lig_names)
            _lig_template[code] = _ligand_template(code)
            _lig_pos_by_name[code] = {nm: i for i, nm in enumerate(_lig_template[code]["names"])}

    # ref_space_uid group id (F4-generalized): a GLOBAL count of distinct
    # residue-groups in first-appearance order (biotite's residue-level
    # indexing) — a protein/NA token is always its own group (1 token == 1
    # residue); ALL of one ligand INSTANCE's per-atom tokens share ONE group
    # (keyed by its CCD code, since one instance's tokens are contiguous and
    # this pass allows only one instance per code). See module docstring
    # (F4 grounding) for why this replaced the old "first ligand token index"
    # framing once a SECOND ligand instance is present.
    _group_id_by_key: dict = {}
    _next_group_id = 0

    pos = 0
    for ti, (tk, kind, (names, coord, tk_is_virtual, elements)) in enumerate(zip(tokens, token_kind, layouts)):
        n = len(names)
        s, e = pos, pos + n
        atom_names.extend(names)
        atom_elements.extend(elements if elements is not None else [None] * n)
        is_virtual[s:e] = tk_is_virtual
        fixed_coord = tk.fixed_coord
        fixed_seq = tk.fixed_seq
        if kind == "protein":
            has_cb = "CB" in names
            for j, nm in enumerate(names):
                if nm in BACKBONE_NAMES:
                    is_backbone[s + j] = True
                    if nm == "CA":
                        is_ca[s + j] = True
                else:
                    is_sidechain[s + j] = True
                    if nm == "CB":
                        is_central[s + j] = True
            if tk.res_name == "GLY" and "CA" in names:
                # GLY has no CB in the dense scheme at all -- the reference's
                # `get_af3_token_representative_masks` has a DEDICATED
                # glycine branch (CA is always its representative), unlike
                # any other residue -- verified unconditional, not just an
                # atomize fallback (see the `is_unindexed` branch below).
                is_central[s + names.index("CA")] = True
            if tk.is_unindexed and n and not is_central[s:e].any():
                # F4: `select_fixed_atoms` can subset an unindexed residue's
                # real atoms to a set with no CB (and not GLY) -- verified vs
                # a real reference capture (enzyme_design.md's M0255_1mg5
                # catalytic residues, e.g. A108 subsetted to {ND2,CG}): the
                # reference then forces the FIRST KEPT atom to be its own
                # representative (`add_representative_atom` -> per-atom
                # `atomize`), the same "atomized" convention already used for
                # ligand atoms -- is_backbone/is_central True, is_sidechain
                # False, for that one atom only (see module docstring).
                is_backbone[s] = True
                is_central[s] = True
                is_sidechain[s] = False
        elif kind == "ligand":
            # A single-atom token IS its own representative atom (verified vs
            # a real reference capture: is_ca/is_central/is_backbone all True,
            # is_sidechain False, for every ligand atom — the AF3 "atomized
            # token" convention, distinct from protein/NA's multi-atom tokens).
            is_ligand_atom[s:e] = True
            is_ca[s:e] = True
            is_central[s:e] = True
            is_backbone[s:e] = True
            # fixed_coord already resolved onto `tk.is_fixed_coord` in
            # `_plan_ligand_tokens` (F4 fix — see its docstring); `fixed_coord`
            # (set from `tk.fixed_coord` a few lines above) is used as-is.
        else:  # dna/rna: never backbone/sidechain-flagged; representative
            # atom is the base ring-center (C4 purine / C2 pyrimidine) —
            # verified vs a real reference capture (see module docstring).
            is_na_atom[s:e] = True
            central = _central_atom_name(tk.res_name)
            if central is not None and central in names:
                j = names.index(central)
                is_ca[s + j] = True
                is_central[s + j] = True
        if fixed_coord:
            is_motif_atom_fixed_coord[s:e] = True
            atom_coord[s:e] = coord
            motif_pos[s:e] = coord - com
        if fixed_seq:
            is_motif_atom_fixed_seq[s:e] = True
        if tk.is_unindexed:
            is_motif_atom_unindexed[s:e] = True
            # Reference override for unindexed tokens: `is_ca` is forced onto
            # the token's FIRST atom regardless of its real name (design_
            # transforms.py: "Ensure is_ca represents one and the first atom
            # only for unindexed tokens") — verified vs a real capture.
            is_ca[s:e] = False
            if n:
                is_ca[s] = True
        # ref_space_uid is a RESIDUE-level index (biotite's get_residue_starts
        # grouping), not a token-level one: a ligand's per-atom tokens all
        # belong to the SAME underlying residue, so they all share ONE group
        # id rather than each atom (or each token) claiming its own —
        # verified vs a real reference capture (see module docstring, F4
        # grounding, for why this is a GLOBAL group counter and not simply
        # "this ligand's own first token index" once 2+ ligand instances are
        # present). Keyed by real residue IDENTITY (chain, res_id), NOT CCD
        # code (p20 fix, same coincidental-until-tested bug as
        # `residue_index` above — see module docstring's "ligand + symmetry"
        # grounding).
        gkey = ("ligand", tk.residue.chain, tk.residue.res_id) if tk.is_ligand else ("single", ti)
        if gkey not in _group_id_by_key:
            _group_id_by_key[gkey] = _next_group_id
            _next_group_id += 1
        ref_space_uid[s:e] = _group_id_by_key[gkey]
        atom_to_token_map[s:e] = ti
        token_res_name.append("UNK" if tk.is_ligand else (tk.res_name or "GAP"))
        token_chain.append(tk.chain)
        token_res_id.append(tk.res_id)
        token_is_motif[ti] = tk.is_motif
        token_is_unindexed[ti] = tk.is_unindexed
        token_break_before[ti] = tk.is_chain_break_before
        token_is_fully_fixed_coord[ti] = bool(is_motif_atom_fixed_coord[s:e].all()) if e > s else False
        pos = e

    # --- reference-conformer features: ALL-ZERO/False for protein (motif or
    # designed alike) — CreateDesignReferenceFeatures.has_sequence excludes
    # protein entirely under generate_conformers_for_non_protein_only, so
    # ref_pos/ref_charge/ref_pos_is_ground_truth never get filled for protein;
    # real motif geometry flows only through motif_pos. For NA (not excluded),
    # ref_mask=True and ref_element is the real per-atom atomic-number one-hot
    # (verified vs real reference captures — see module docstring); ref_pos
    # stays 0 (the real reference-conformer 3D geometry needs RDKit/CCD
    # embedding this port does not vendor — documented gap) and ref_charge
    # stays 0 (matches both real captures: no formally-charged atoms in the
    # standard-nucleotide neutral conformer).
    ref_pos = np.zeros((L, 3), dtype=np.float32)
    ref_mask = np.array(is_na_atom | is_ligand_atom, dtype=bool)
    ref_pos_is_ground_truth = np.zeros(L, dtype=bool)
    ref_charge = np.zeros(L, dtype=np.int8)
    ref_element = np.zeros((L, 128), dtype=np.float32)
    ref_element[:, 0] = 1.0
    for i, (elem, is_na) in enumerate(zip(atom_elements, is_na_atom)):
        if is_na and elem in _ELEMENT_TO_ATOMIC_NUMBER:
            ref_element[i, 0] = 0.0
            ref_element[i, _ELEMENT_TO_ATOMIC_NUMBER[elem]] = 1.0
    ref_atom_name_chars = _encode_atom_names_like_af3(atom_names)  # [L,4,64] f32 (live-pipeline dtype)
    has_zero_occupancy = np.zeros(L, dtype=bool)  # forced False at inference regardless of input

    # atomworks-ungated defaults (all-zero in the default inference config —
    # verified vs a real reference capture: FeaturizeAtoms' rasa_binned default
    # bin is excluded from the one-hot; no hbond/hotspot annotation present).
    ref_atomwise_rasa = np.zeros((L, 3), dtype=np.int64)
    active_donor = np.zeros(L, dtype=np.int64)
    active_acceptor = np.zeros(L, dtype=np.int64)
    is_atom_level_hotspot = np.zeros((L, 1), dtype=np.float32)

    # --- ligand (F3/F4) reference-conformer + RASA-conditioning features ---
    # ref_pos/ref_charge/ref_element come from the real CCD template (matched
    # by real atom name, verified vs a real reference capture: element/charge
    # ARE real per-atom values for a ligand, unlike protein's always-zero
    # placeholder). ref_atom_name_chars is overridden to encode the ELEMENT
    # symbol, not the real atom name — `use_element_for_atom_names_of_
    # atomized_tokens=True` in the reference's default inference config
    # (verified: a real capture's ligand rows decode to "C   "/"N   ", not
    # "C22 "/"N9  "). select_buried/select_exposed become the one-hot
    # `ref_atomwise_rasa` bin DIRECTLY (bin 0=buried, 2=exposed) — this is a
    # user-specified per-atom label at inference, NOT a computed SASA value
    # (rfd3.inference.input_parsing assigns `rasa_bin` straight from the
    # selection; `rfd3.transforms.rasa.CalculateRASA`'s real Shrake-Rupley
    # computation is training-only, never invoked at inference).
    if spec.ligand:
        # F4: track each CODE's own position-within-its-instance counter so a
        # second (or third) ligand instance's buried/exposed mask indexes
        # correctly, even though atoms of DIFFERENT codes are interleaved in
        # `lig_atom_idx` only in the sense that each code's own run is
        # contiguous (never interleaved with another code's atoms).
        _lig_pos_ctr: dict = {}
        lig_atom_idx = np.where(is_ligand_atom)[0]
        for atom_i in lig_atom_idx:
            ti = int(atom_to_token_map[atom_i])
            code = tokens[ti].res_name
            k = _lig_pos_ctr.get(code, 0)
            _lig_pos_ctr[code] = k + 1
            nm = atom_names[atom_i]
            tmpl_i = _lig_pos_by_name[code].get(nm)
            if tmpl_i is not None:
                ref_pos[atom_i] = _lig_template[code]["coord"][tmpl_i]
                ref_charge[atom_i] = _lig_template[code]["charges"][tmpl_i]
                elem = _lig_template[code]["elements"][tmpl_i]
                if elem in _ELEMENT_TO_ATOMIC_NUMBER:
                    ref_element[atom_i, 0] = 0.0
                    ref_element[atom_i, _ELEMENT_TO_ATOMIC_NUMBER[elem]] = 1.0
                ref_atom_name_chars[atom_i] = _encode_atom_names_like_af3([elem])[0]
            if _lig_buried_mask[code][k]:
                ref_atomwise_rasa[atom_i] = [1, 0, 0]
            elif _lig_exposed_mask[code][k]:
                ref_atomwise_rasa[atom_i] = [0, 0, 1]

    # --- token-level features ---
    restype = _restype_onehot(token_res_name).astype(np.int64)  # [I,32] one-hot int64
    motif_token_class = np.zeros(I, dtype=np.int8)
    motif_token_class[token_is_motif] = 1
    motif_token_class[token_is_unindexed] = 2
    ref_motif_token_type = np.eye(3, dtype=np.int8)[motif_token_class]
    ref_plddt = np.where(token_is_motif, 0, 1).astype(np.int64)
    # is_non_loopy: a per-spec "temperature conditioning" toggle
    # (`spec.is_non_loopy`), applied only to DIFFUSED (non-motif) tokens --
    # verified against the real reference (`input_parsing.py::_apply_globals`:
    # unset -> 0 everywhere; set -> 1/-1 on `~is_motif_token`, still 0 on any
    # motif/unindexed token). Pre-existing bug found by F5 (p18): this was
    # hardcoded to all-zero regardless of `spec.is_non_loopy`, uncaught
    # because no F1-F4 fixture ever set the field (see module docstring).
    is_non_loopy = np.zeros((I, 1), dtype=np.float32)
    if spec.is_non_loopy is not None:
        is_non_loopy[~token_is_motif, 0] = 1.0 if spec.is_non_loopy else -1.0
    is_motif_token_unindexed = token_is_unindexed.copy()
    is_motif_token_with_fully_fixed_coord = token_is_fully_fixed_coord

    is_protein_tok = np.array([k == "protein" for k in token_kind], dtype=bool)
    is_rna_tok = np.array([k == "rna" for k in token_kind], dtype=bool)
    is_dna_tok = np.array([k == "dna" for k in token_kind], dtype=bool)
    is_ligand_tok = np.array([k == "ligand" for k in token_kind], dtype=bool)
    _POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "LYS", "ARG", "ASP", "GLU"}
    is_polar = is_protein_tok & np.isin(np.array(token_res_name), np.array(list(_POLAR)))

    is_N_term = np.zeros(I, dtype=bool)
    is_C_term = np.zeros(I, dtype=bool)
    for ti in range(I):
        first_seg = (ti == 0) or token_break_before[ti] or (token_chain[ti] != token_chain[ti - 1])
        is_N_term[ti] = first_seg
        # The real chain's C-terminus lands on the last non-unindexed token
        # even though the array continues with the appended unindexed block
        # (verified vs a real reference capture: the token right before the
        # first unindexed token IS flagged C-terminus).
        entering_unindexed = token_is_unindexed[ti + 1] and not token_is_unindexed[ti] if ti + 1 < I else False
        last_in_seg = (ti == I - 1) or token_break_before[ti + 1] or (token_chain[ti + 1] != token_chain[ti]) or entering_unindexed
        is_C_term[ti] = last_in_seg
    # Unindexed tokens never carry a terminus flag (verified vs a real
    # reference capture: terminus_type is all-zero for every unindexed token,
    # regardless of island boundaries — add_protein_termini_annotation is not
    # re-applied to them). Same for NA (5'/3' ends): terminus_type is
    # protein-only in the reference contract — verified all-zero for DNA/RNA
    # on both real captures.
    is_N_term[token_is_unindexed | ~is_protein_tok] = False
    is_C_term[token_is_unindexed | ~is_protein_tok] = False
    terminus_type = np.zeros((I, 2), dtype=np.int64)
    terminus_type[is_C_term, 0] = 1
    terminus_type[is_N_term, 1] = 1

    chain_to_asym = {}
    for c in token_chain:
        if c not in chain_to_asym:
            chain_to_asym[c] = len(chain_to_asym)  # 0-based
    asym_id = np.array([chain_to_asym[c] for c in token_chain], dtype=np.int64)

    # entity_id/sym_id: chains sharing the SAME full real-chain residue-name
    # sequence (not just the contig-selected subset) are the same entity,
    # with sym_id enumerating replica copies — verified vs a real reference
    # capture (dsDNA_basic: chain A and B are the same 12-mer palindrome, so
    # they share entity_id with distinct sym_id). A synthetic (designed)
    # chain has no real sequence to match and always starts a fresh entity —
    # EXCEPT F5's symmetric replicas, which share no real sequence either but
    # must still share ONE entity_id (verified vs a real reference capture: 3
    # C3 replicas of a 12-residue unconditional design all get entity_id=0,
    # not 3 distinct entities). `sym_chains` (F5 only, empty otherwise) is the
    # override for exactly that case; `sym_group_entity_id` is assigned once,
    # the first time a sym_chains member is seen (in token order, i.e. the
    # ASU itself, so the shared entity lands wherever the ASU's own chain
    # would have — the downstream sym_id loop below then naturally enumerates
    # 0..N-1 in ASU-then-replica order since it iterates the same insertion
    # order).
    chain_full_seq: dict[str, tuple] = {}
    for r in residues:
        chain_full_seq.setdefault(r.chain, []).append((r.res_id, r.res_name))
    chain_full_seq = {c: tuple(name for _, name in sorted(v)) for c, v in chain_full_seq.items()}
    entity_of_seq: dict[tuple, int] = {}
    chain_entity: dict[str, int] = {}
    next_entity = 0
    sym_group_entity_id: int | None = None
    ligand_group_entity_id: int | None = None
    for c in chain_to_asym:  # insertion order == order of first appearance among tokens
        seq = chain_full_seq.get(c)
        if sym_chains and c in sym_chains:
            if sym_group_entity_id is None:
                sym_group_entity_id = next_entity
                next_entity += 1
            eid = sym_group_entity_id
        elif ligand_chains and c in ligand_chains:
            # F4/F5: every ligand chain -- one shared chain for a
            # non-symmetric input, one PER real source subunit for a
            # symmetric one (see `_plan_ligand_tokens`) -- shares ONE
            # entity_id, mirroring `sym_group_entity_id`'s "no real
            # sequence, but still one group" override (verified vs a real
            # reference capture: a symmetric enzyme's two per-subunit ligand
            # chains both land on the SAME entity, not two distinct ones;
            # for a single ligand chain this reduces to exactly the
            # pre-existing "give it a fresh entity" behavior below, zero
            # regression).
            if ligand_group_entity_id is None:
                ligand_group_entity_id = next_entity
                next_entity += 1
            eid = ligand_group_entity_id
        elif seq is None or seq not in entity_of_seq:
            eid = next_entity
            next_entity += 1
            if seq is not None:
                entity_of_seq[seq] = eid
        else:
            eid = entity_of_seq[seq]
        chain_entity[c] = eid
    entity_id = np.array([chain_entity[c] for c in token_chain], dtype=np.int64)
    sym_counter: dict[int, int] = {}
    chain_sym: dict[str, int] = {}
    for c in chain_to_asym:
        e = chain_entity[c]
        chain_sym[c] = sym_counter.get(e, 0)
        sym_counter[e] = chain_sym[c] + 1
    sym_id = np.array([chain_sym[c] for c in token_chain], dtype=np.int32)
    residue_index = np.zeros(I, dtype=np.int32)
    _per_chain_ctr = {}
    _ligand_res_idx_by_key = {}  # (chain, real source chain, real res_id) -> the one residue_index shared by that INSTANCE's atoms
    for ti, c in enumerate(token_chain):
        if tokens[ti].is_ligand:
            # All of ONE ligand instance's per-atom tokens are the SAME
            # underlying residue -> they share ONE residue_index; every
            # OTHER instance on that chain -- a different code (F4:
            # enzyme_design.md's NAI -> 0, ACT -> 1) OR a second instance of
            # the SAME code (p20: 1j79's two Zn on one subunit's ligand
            # chain -> 1, 2) -- consumes its OWN slot in that chain's
            # counter. Keyed by real residue IDENTITY (chain, res_id), NOT
            # by CCD code (p20 fix — see module docstring's "ligand +
            # symmetry" grounding: code-keying was only coincidentally right
            # while every case had exactly one instance per code).
            key = (c, tokens[ti].residue.chain, tokens[ti].residue.res_id)
            if key not in _ligand_res_idx_by_key:
                _ligand_res_idx_by_key[key] = _per_chain_ctr.get(c, 0)
                _per_chain_ctr[c] = _ligand_res_idx_by_key[key] + 1
            residue_index[ti] = _ligand_res_idx_by_key[key]
        else:
            residue_index[ti] = _per_chain_ctr.get(c, 0)  # 0-based per chain
            _per_chain_ctr[c] = residue_index[ti] + 1
    token_index = np.arange(I, dtype=np.int64)

    # token_bonds: ALL FALSE for standard contiguous protein (not the peptide-
    # bond graph — encodes inter-token bonds for modified residues/crosslinks/
    # ligands only). A ligand IS one such case: since each atom is its own
    # token, the real intra-ligand covalent bond graph (from the CCD template)
    # becomes real inter-TOKEN bonds — verified vs a real reference capture
    # (a 33-heavy-atom ligand token block has a real, non-trivial token_bonds
    # submatrix, not all-zero). F4: each ligand INSTANCE's bond graph is
    # resolved separately, keyed by (code, real residue identity) — NOT just
    # by code (p20 fix: a plain per-code `name_to_tok` dict would silently
    # collide two instances of the SAME code, since both share atom names —
    # the SECOND instance's token indices would overwrite the FIRST's, wiring
    # every bond onto the last instance only and leaving earlier instances
    # bond-free; verified vs 1j79's two ORO instances, real ring atoms with
    # real intra-residue bonds). Different-code instances sharing a chain
    # already never cross-wire each other's bonds (verified: zero
    # cross-ligand token_bonds entries in a real capture of "NAI,ACT") since
    # each instance's dict is built fresh.
    token_bonds = np.zeros((I, I), dtype=bool)
    if spec.ligand:
        _lig_instances: dict[tuple, dict[str, int]] = {}
        for ti in np.where(is_ligand_tok)[0]:
            tk = tokens[ti]
            _lig_instances.setdefault((tk.res_name, tk.residue.chain, tk.residue.res_id), {})[
                tk.ligand_atom_name] = ti
        for (code, _src_chain, _src_res_id), name_to_tok in _lig_instances.items():
            for u, v in _lig_template[code]["bonds"]:
                tu = name_to_tok.get(_lig_template[code]["names"][u])
                tv = name_to_tok.get(_lig_template[code]["names"][v])
                if tu is not None and tv is not None:
                    token_bonds[tu, tv] = token_bonds[tv, tu] = True

    # unindexing_pair_mask: True = RPE must NOT leak relative position between
    # this token pair (UnindexFlaggedTokens.create_unindexed_masks). Indexed<->
    # unindexed is ALWAYS masked; unindexed<->unindexed is masked unless the
    # two tokens are in the same "island" (contiguous '-' range in the unindex
    # spec). Verified vs a real reference capture (scripts/rfd3_port/
    # parity_artifacts/parity_unindex.py).
    unindexing_pair_mask = np.zeros((I, I), dtype=bool)
    ui = token_is_unindexed
    if ui.any():
        unindexing_pair_mask = ui[:, None] ^ ui[None, :]
        idx_ui = np.where(ui)[0]
        island = np.cumsum([tokens[i].unindex_new_island for i in idx_ui])
        sub_mask = island[:, None] != island[None, :]
        unindexing_pair_mask[np.ix_(idx_ui, idx_ui)] = sub_mask

    # F5 symmetry: broadcast each token's transform id to its atoms, plus the
    # ASU mask + the raw (R,t) transform dict the sampler needs to actually
    # re-derive every replica's coordinates from the ASU each step (see
    # module docstring's F5 grounding + tt_bio.rfd3_sampler). sym_entity_id is
    # 0 for every REPLICATED atom (this pass's single symmetric protein
    # group); a token forced to transform_id=-1 (is_unsym_motif, or an
    # unindexed motif regardless of which replica it came from — see the
    # module docstring's F5+motif grounding, mechanisms (b)/(c)) gets the
    # reference's FIXED_ENTITY_ID=-1 sentinel too, so the sampler's existing
    # `apply_symmetry_atomwise` never resymmetrizes it.
    sym_transform = None
    if sym_frames is not None:
        tok_transform_id = np.asarray(sym_transform_id_by_token, dtype=np.int32)
        sym_transform_id = tok_transform_id[atom_to_token_map]
        sym_entity_id = np.where(sym_transform_id == -1, -1, 0).astype(np.int64)
        is_sym_asu = sym_transform_id == 0
        sym_transform = {
            str(i): (torch.from_numpy(R), torch.from_numpy(t))
            for i, (R, t) in enumerate(sym_frames)
        }

    bf = lambda a: torch.from_numpy(a)
    f = {
        # atom-level
        "ref_atom_name_chars": bf(ref_atom_name_chars),                # [L,4,64] f32
        "ref_pos": bf(ref_pos),                                        # [L,3] f32
        "ref_mask": torch.from_numpy(ref_mask),                        # [L] bool
        "ref_element": bf(ref_element),                                # [L,128] f32
        "ref_charge": torch.from_numpy(ref_charge),                    # [L] int8
        "ref_space_uid": torch.from_numpy(ref_space_uid),              # [L] int64
        "ref_pos_is_ground_truth": torch.from_numpy(ref_pos_is_ground_truth),  # [L] bool
        "has_zero_occupancy": torch.from_numpy(has_zero_occupancy),     # [L] bool
        "ref_is_motif_atom_with_fixed_coord": torch.from_numpy(is_motif_atom_fixed_coord),
        "ref_is_motif_atom_unindexed": torch.from_numpy(is_motif_atom_unindexed),
        "ref_atomwise_rasa": torch.from_numpy(ref_atomwise_rasa),      # [L,3] int64
        "active_donor": torch.from_numpy(active_donor),                # [L] int64
        "active_acceptor": torch.from_numpy(active_acceptor),          # [L] int64
        "is_atom_level_hotspot": bf(is_atom_level_hotspot),            # [L,1] f32
        "is_motif_atom_with_fixed_coord": torch.from_numpy(is_motif_atom_fixed_coord),
        "is_motif_atom_with_fixed_seq": torch.from_numpy(is_motif_atom_fixed_seq),
        "is_motif_atom_unindexed": torch.from_numpy(is_motif_atom_unindexed),
        "motif_pos": bf(motif_pos),                                  # [L,3] f32
        "is_ca": torch.from_numpy(is_ca),
        "is_central": torch.from_numpy(is_central),
        "is_backbone": torch.from_numpy(is_backbone),
        "is_sidechain": torch.from_numpy(is_sidechain),
        "is_virtual": torch.from_numpy(is_virtual),
        "atom_to_token_map": torch.from_numpy(atom_to_token_map),     # [L] int32
        # token-level
        "restype": torch.from_numpy(restype),                          # [I,32] int64 one-hot
        "ref_motif_token_type": torch.from_numpy(ref_motif_token_type),  # [I,3] int8 one-hot
        "ref_plddt": torch.from_numpy(ref_plddt),                      # [I] int64
        "is_non_loopy": bf(is_non_loopy),                              # [I,1] f32
        "is_motif_token_unindexed": torch.from_numpy(is_motif_token_unindexed),  # [I] bool
        "is_motif_token_with_fully_fixed_coord": torch.from_numpy(is_motif_token_with_fully_fixed_coord),
        "is_protein": torch.from_numpy(is_protein_tok),                # [I] bool
        "is_rna": torch.from_numpy(is_rna_tok),                        # [I] bool
        "is_dna": torch.from_numpy(is_dna_tok),                        # [I] bool
        "is_ligand": torch.from_numpy(is_ligand_tok),                  # [I] bool
        "is_polar": torch.from_numpy(is_polar),                        # [I] bool
        "terminus_type": torch.from_numpy(terminus_type),              # [I,2] int64 one-hot
        "asym_id": torch.from_numpy(asym_id),
        "entity_id": torch.from_numpy(entity_id),
        "sym_id": torch.from_numpy(sym_id),                          # [I] int32
        "residue_index": torch.from_numpy(residue_index),              # [I] int32
        "token_index": torch.from_numpy(token_index),
        "token_bonds": torch.from_numpy(token_bonds),                  # [I,I] bool
        "unindexing_pair_mask": torch.from_numpy(unindexing_pair_mask),
    }
    if sym_transform is not None:
        f["sym_transform"] = sym_transform                            # {str(id): (R[3,3], t[3])}
        f["sym_transform_id"] = torch.from_numpy(sym_transform_id)     # [L] int32
        f["sym_entity_id"] = torch.from_numpy(sym_entity_id)           # [L] int64
        f["is_sym_asu"] = torch.from_numpy(is_sym_asu)                 # [L] bool
    return f
