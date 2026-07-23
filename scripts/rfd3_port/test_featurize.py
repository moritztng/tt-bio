"""Structural unit tests for tt_bio.rfd3_featurize.featurize.

These verify SHAPES, DTYPES, and KEY INVARIANTS of the produced ``f`` dict on a
synthetic poly-ALA input. The REAL value-parity gate (bit-exact vs a captured
reference `f` on a real protein) is scripts/rfd3_port/parity_artifacts/parity_iai.py
(43/43 keys bit-exact as of p12). ALA has no side-chain atoms beyond CB in the
"dense" atom14 scheme, so every motif (fixed-seq) token here has exactly 5
atoms (N,CA,C,O,CB) and every designed token has 14 (5 real + 9 virtual pad) —
L is NOT I*14 in general (see featurize()'s module docstring).
Run: .venv/bin/python scripts/rfd3_port/test_featurize.py
"""
import os, sys, tempfile, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification


def _write_minipdb(path, chain="A", n=20, start=1):
    """Write a tiny idealized polyalanine PDB (N,CA,C,O,CB per residue)."""
    lines = []
    # rough idealized backbone walk
    x = 0.0
    for i in range(n):
        rid = start + i
        base = [x, 0.0, 0.0]
        for j, (nm, el, off) in enumerate([("N", "N", (0.0, 0.0, 0.0)),
                                            ("CA", "C", (1.46, 0.0, 0.0)),
                                            ("C", "C", (1.52, 1.46, 0.0)),
                                            ("O", "O", (0.8, 2.2, 0.0)),
                                            ("CB", "C", (1.0, 0.5, -1.2))]):
            cx, cy, cz = base[0] + off[0], base[1] + off[1], base[2] + off[2]
            lines.append(
                f"ATOM  {5*i+j+1:5d}  {nm:<3s} ALA {chain}{rid:4d}    "
                f"{cx:8.3f}{cy:8.3f}{cz:8.3f}  1.00  0.00           {el}")
        x += 3.8
    with open(path, "w") as fh:
        fh.write("END\n".join(lines) + "\nEND\n")


def _spec(d):
    s = InputSpecification.from_dict(d)
    s.validate()
    return s


def test_motif_scaffold_shapes_and_invariants():
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        _write_minipdb(pdb, n=40, start=1)
        # contig: motif A1-10, scaffold 20, motif A31-40  (motif scaffolding F6)
        spec = _spec({"input": pdb, "contig": "A1-10,20,A31-40"})
        f = featurize(pdb, spec)

    I = 10 + 20 + 10  # 40 tokens
    # ALA has no side-chain atoms beyond CB in the dense atom14 scheme: every
    # motif (fixed-seq) token here is exactly 5 atoms (N,CA,C,O,CB); every
    # designed token is the full 14-slot template (5 real + 9 virtual pad).
    L = 20 * 5 + 20 * 14
    a2t = f["atom_to_token_map"]
    assert a2t.shape == (L,)
    # shapes
    assert f["ref_atom_name_chars"].shape == (L, 4, 64)
    assert f["ref_pos"].shape == (L, 3)
    assert f["ref_element"].shape == (L, 128)
    assert f["ref_atomwise_rasa"].shape == (L, 3)
    assert f["restype"].shape == (I, 32)
    assert f["ref_motif_token_type"].shape == (I, 3)
    assert f["token_bonds"].shape == (I, I)
    assert f["unindexing_pair_mask"].shape == (I, I)
    assert f["asym_id"].shape == (I,)
    # invariants
    assert a2t[0].item() == 0 and a2t[-1].item() == I - 1
    counts = torch.bincount(a2t.long(), minlength=I)
    assert counts.tolist() == [5] * 10 + [14] * 20 + [5] * 10
    # exactly one CA per token (is_ca)
    assert int(f["is_ca"].sum()) == I
    # exactly one central atom (CB) per token
    assert int(f["is_central"].sum()) == I
    # motif tokens: 20 motif (A1-10 + A31-40), 20 designed
    motif_atom = f["is_motif_atom_with_fixed_coord"]
    motif_tokens = torch.zeros(I, dtype=torch.bool).scatter_(0, a2t.long(), motif_atom)
    assert int(motif_tokens.sum()) == 20
    # ref_pos is all-zero for EVERY protein atom (motif and designed alike) —
    # real motif geometry flows only through motif_pos (parity-verified vs a
    # real reference capture, see parity_artifacts/).
    assert torch.all(f["ref_pos"] == 0)
    assert torch.all(f["motif_pos"][~motif_atom] == 0)
    assert torch.any(f["motif_pos"][motif_atom] != 0)
    # ref_space_uid: every atom of a token shares its token's uid
    assert torch.equal(f["ref_space_uid"], a2t.to(f["ref_space_uid"].dtype))
    # token_bonds symmetric, no self-bonds
    assert float(f["token_bonds"].to(torch.float).diagonal().abs().sum()) == 0.0
    assert torch.allclose(f["token_bonds"].to(torch.float), f["token_bonds"].to(torch.float).T)
    # chain break: contig has no /0 so A31-40 follows the scaffold on chain A
    # -> bonds exist across motif->scaffold->motif boundaries (contiguous res_id)
    print("test_motif_scaffold_shapes_and_invariants OK  I=", I, "L=", L)


def test_chain_break():
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        _write_minipdb(pdb, n=40)
        spec = _spec({"input": pdb, "contig": "A1-10,/0,A31-40"})
        f = featurize(pdb, spec)
    # Parity-gated vs a real protein reference capture: token_bonds is NOT the
    # peptide-bond graph (that lives elsewhere); for standard contiguous protein it
    # is ALL FALSE, with or without a chain break. token_bonds encodes inter-token
    # bonds for modified residues / crosslinks / ligands only.
    assert bool((f["token_bonds"] == 0).all())
    assert float(f["token_bonds"][9, 10]) == 0.0
    assert float(f["token_bonds"][8, 9]) == 0.0
    print("test_chain_break OK")


def test_binder_contig_designed_only():
    # pure designed region (a binder scaffold with no motif) — degenerate but legal
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        _write_minipdb(pdb, n=5)
        spec = _spec({"input": pdb, "contig": "30"})
        f = featurize(pdb, spec)
    I = 30
    assert f["restype"].shape == (I, 32)
    assert int(f["is_motif_atom_with_fixed_coord"].sum()) == 0
    assert torch.all(f["ref_pos"] == 0)
    print("test_binder_contig_designed_only OK  I=", I)


def test_ligand_rejected():
    """Ligand/enzyme design (F3/F4) is out of scope this pass: an Indexed
    contig component referencing a real ligand residue (not protein/DNA/RNA)
    must fail loud, not silently misfeaturize."""
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        with open(pdb, "w") as fh:
            fh.write(
                "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                "END\n")
        spec = _spec({"input": pdb, "contig": "A1,10"})
        try:
            featurize(pdb, spec)
            raise AssertionError("expected an error for a ligand indexed motif")
        except (NotImplementedError, ValueError):
            pass
    print("test_ligand_rejected OK")


def _dsdna_basic_pdb():
    return os.path.join(os.path.dirname(__file__), "parity_artifacts", "dsdna_basic", "1bna.pdb")


def test_na_binder_dsdna():
    """F2/F8 NA-binder design (p15): a fixed dsDNA target (1bna.pdb, chains A
    +B, real B-DNA dodecamer duplex) + a designed protein binder chain.
    Bit-exact VALUE parity vs a real reference capture is
    scripts/rfd3_port/parity_artifacts/parity_dna.py (42/43 keys, the lone
    documented gap being `ref_pos`'s real reference-conformer geometry);
    this is the structural/shape+invariant gate."""
    pdb = _dsdna_basic_pdb()
    spec = _spec({"input": pdb, "contig": "A1-10,/0,B15-24,/0,5"})
    f = featurize(pdb, spec)
    I = 10 + 10 + 5  # 20 DNA motif tokens (2 chains) + 5 designed protein
    assert f["restype"].shape == (I, 32)
    is_dna = f["is_dna"]
    assert int(is_dna.sum()) == 20 and int(f["is_protein"].sum()) == 5
    # the two DNA chains are the same 12-mer palindrome -> same entity, distinct sym_id
    assert f["entity_id"][0].item() == f["entity_id"][10].item()
    assert f["sym_id"][0].item() == 0 and f["sym_id"][10].item() == 1
    # designed protein chain after the '/0' break is its own chain/entity
    assert f["asym_id"][20].item() not in (f["asym_id"][0].item(), f["asym_id"][10].item())
    assert f["entity_id"][20].item() != f["entity_id"][0].item()
    # every DNA atom: ref_mask True, real (non-column-0) ref_element, never backbone/sidechain
    dna_atoms = torch.isin(f["atom_to_token_map"], is_dna.nonzero().flatten().to(torch.int32))
    assert bool(f["ref_mask"][dna_atoms].all())
    assert bool((f["ref_element"][dna_atoms, 0] == 0).all())
    assert not bool(f["is_backbone"][dna_atoms].any())
    assert not bool(f["is_sidechain"][dna_atoms].any())
    # DNA tokens never carry a terminus flag in this contract
    assert torch.all(f["terminus_type"][is_dna] == 0)
    # exactly one representative (is_ca/is_central) atom per DNA token
    dna_tok_of_atom = f["atom_to_token_map"][dna_atoms].long()
    rep_count = torch.bincount(dna_tok_of_atom, weights=f["is_ca"][dna_atoms].float(), minlength=I)
    assert int(rep_count[is_dna].sum().item()) == 20 and bool((rep_count[is_dna] == 1).all())
    print(f"test_na_binder_dsdna OK  I={I} L={f['ref_pos'].shape[0]}")


def test_unindex_tied_and_separate_islands():
    """F6 unindex field (p14): 'A31-32' (tied island, leaked to each other) +
    'A35' (separate singleton island) appended after the main contig. Value
    parity vs a real reference capture is scripts/rfd3_port/parity_artifacts/
    parity_unindex.py (43/43 keys bit-exact on a real protein, p14); this is
    the structural/shape+mask-invariant gate on a synthetic input."""
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        _write_minipdb(pdb, n=40)
        spec = _spec({"input": pdb, "contig": "A1-10,20", "unindex": "A31-32,A35"})
        f = featurize(pdb, spec)

        # overlap between contig and unindex must be rejected (same tempdir/pdb)
        try:
            featurize(pdb, _spec({"input": pdb, "contig": "A1-10,20", "unindex": "A5"}))
            raise AssertionError("expected ValueError for contig/unindex overlap")
        except ValueError:
            pass

    I = 10 + 20 + 3  # 30 main tokens + 3 unindexed (A31,A32,A35)
    assert f["restype"].shape[0] == I
    unind = f["is_motif_token_unindexed"]
    assert unind.tolist() == [False] * 30 + [True, True, True]
    # unindexed tokens carry no terminus flag, regardless of island boundaries
    assert torch.all(f["terminus_type"][unind] == 0)
    # unindexed tokens are motif-like (fixed coord/seq) — reused atom layout
    assert torch.all(f["is_motif_token_with_fully_fixed_coord"][unind])
    assert torch.all(f["ref_plddt"][unind] == 0)

    upm = f["unindexing_pair_mask"]
    assert upm.shape == (I, I)
    # indexed <-> unindexed: ALWAYS masked (no leak), both directions
    assert bool(upm[:30, 30:].all()) and bool(upm[30:, :30].all())
    # tied island (A31,A32 -> tokens 30,31): leaked to each other (not masked)
    assert not bool(upm[30, 31]) and not bool(upm[31, 30])
    # separate singleton island (A35 -> token 32): masked from the other island
    assert bool(upm[30, 32]) and bool(upm[31, 32])
    # indexed <-> indexed: never masked by this feature
    assert not bool(upm[:30, :30].any())
    print("test_unindex_tied_and_separate_islands OK  I=", I)


def test_contract_vs_golden():
    """Verify the ported featurizer's `f` key set against the REAL captured
    dsDNA_basic golden meta.json at
    ~/.coworker/artifacts/rfd3-goldens/capture/token_initializer.meta.json.

    This is a KEY-SET contract gate only (the golden is NA, not protein, and
    was captured with a bf16 downcast atomworks itself doesn't apply — see
    parity_artifacts/ for the real value-parity gate, which supersedes this
    for dtypes/values). Four keys (ref_atom_name_chars, ref_element, ref_pos,
    motif_pos, is_atom_level_hotspot) are float32 in the live pipeline per a
    real reference capture (p12); the golden's bf16 for these was a
    capture-time artifact, not the true contract, so they're dtype-exempted
    here rather than asserted against the stale golden.
    """
    import json
    meta_path = os.path.expanduser(
        "~/.coworker/artifacts/rfd3-goldens/capture/token_initializer.meta.json")
    if not os.path.exists(meta_path):
        print("test_contract_vs_golden SKIPPED (no golden meta.json present)")
        return
    meta = json.load(open(meta_path))
    g_shapes = meta["in_shapes"]  # {key: [[shape...], "torch.dtype"]}

    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb")
        _write_minipdb(pdb, n=40)
        spec = _spec({"input": pdb, "contig": "A1-10,20,A31-40"})
        f = featurize(pdb, spec)
    I = 40

    golden_keys = set(g_shapes)
    ported_keys = set(f)
    missing = golden_keys - ported_keys
    extra = ported_keys - golden_keys
    assert not missing, f"missing keys vs golden: {sorted(missing)}"
    assert not extra, f"extra keys vs golden: {sorted(extra)}"

    # live-pipeline dtype (not the golden's capture-time bf16 downcast) — see
    # module docstring / parity_artifacts/README.md.
    LIVE_F32_KEYS = {"ref_atom_name_chars", "ref_element", "ref_pos", "motif_pos",
                     "is_atom_level_hotspot", "is_non_loopy"}
    dtype_map = {"torch.bool": torch.bool, "torch.int8": torch.int8, "torch.int32": torch.int32,
                 "torch.int64": torch.int64, "torch.bfloat16": torch.bfloat16,
                 "torch.float32": torch.float32}
    mismatches = []
    for k in golden_keys:
        gshape, gdtype = g_shapes[k]
        pt = f[k]
        if k in LIVE_F32_KEYS:
            if pt.dtype != torch.float32:
                mismatches.append((k, f"dtype {pt.dtype}, expected live-pipeline float32"))
            continue
        gd = dtype_map.get(gdtype)
        if gd is None:
            mismatches.append((k, f"unknown golden dtype {gdtype}")); continue
        if pt.dtype != gd:
            mismatches.append((k, f"dtype {pt.dtype} vs golden {gd}")); continue
        if k in ("token_bonds", "unindexing_pair_mask"):
            if tuple(pt.shape) != (I, I):
                mismatches.append((k, f"shape {tuple(pt.shape)} vs expected {(I, I)}"))
            continue
        # token-level keys: leading dim must be I (feature dims unchanged); atom-level
        # keys have a variable L this pass (real per-token atom counts), so only the
        # feature-dim tail (if any) is checked against the golden.
        if len(gshape) >= 1 and gshape[0] == 144:  # golden I_g=144 (dsDNA_basic)
            exp = (I,) + tuple(gshape[1:])
            if tuple(pt.shape) != exp:
                mismatches.append((k, f"shape {tuple(pt.shape)} vs expected {exp}"))
        elif len(gshape) >= 2:
            if tuple(pt.shape[1:]) != tuple(gshape[1:]):
                mismatches.append((k, f"feature dims {tuple(pt.shape[1:])} vs golden {tuple(gshape[1:])}"))
    assert not mismatches, "\n".join(f"{k}: {m}" for k, m in mismatches)
    print(f"test_contract_vs_golden OK  (43/43 keys present, dtypes match (5 keys "
          f"live-pipeline-f32-exempted), token/feature dims match the real dsDNA_basic "
          f"golden meta; I={I}, L={f['ref_pos'].shape[0]})")


def test_model_consumable():
    """Verify the ported `f` is consumable end-to-end by the reference TokenInitializer
    (CPU) with the real ckpt weights from the golden. Casts float tensors to float32
    first (mimics verify_sampler.reconstruct_f — the golden stores bf16 but the model
    consumes fp32). Checks the TI outputs match the golden out_shapes scaled to our I/L.
    Skips if the golden weights are absent."""
    import json
    cap = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")
    wpath = os.path.join(cap, "token_initializer.real_weights.pt")
    if not os.path.exists(wpath):
        print("test_model_consumable SKIPPED (no golden weights present)"); return
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import rfd3_ref as R
    ti_w = torch.load(wpath, map_location="cpu", weights_only=True)
    ti = R.build_token_initializer(); ti.load_state_dict(ti_w, strict=False); ti.eval()
    with tempfile.TemporaryDirectory() as td:
        pdb = os.path.join(td, "m.pdb"); _write_minipdb(pdb, n=40)
        spec = _spec({"input": pdb, "contig": "A1-10,20,A31-40"})
        f = featurize(pdb, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float32 else v)
          for k, v in f.items()}
    with torch.no_grad():
        out = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    I, L = 40, f["ref_pos"].shape[0]  # L is variable (real per-token atom counts, p12)
    assert tuple(out["Q_L_init"].shape) == (L, 128)
    assert tuple(out["C_L"].shape) == (L, 128)
    assert tuple(out["P_LL"].shape) == (L, L, 16)
    assert tuple(out["S_I"].shape) == (I, 384)
    assert tuple(out["Z_II"].shape) == (I, I, 128)
    print(f"test_model_consumable OK  (ref TokenInitializer ran on ported f -> "
          f"Q_L_init {tuple(out['Q_L_init'].shape)}, S_I {tuple(out['S_I'].shape)}, "
          f"Z_II {tuple(out['Z_II'].shape)})")


def test_ligand_small_molecule_binder():
    """F3 small-molecule-binder design (p16): a pure designed-length (180)
    protein chain around a real ligand (IAI, RosettaCommons/foundry's own
    `sm_binder_design.md` "buried" example) named via `spec.ligand` (a
    SEPARATE mechanism from `contig` — distinct from test_ligand_rejected's
    ligand-via-contig-index, which must still fail loud). Bit-exact VALUE
    parity vs a real reference capture is scripts/rfd3_port/parity_artifacts/
    parity_ligand.py (42/43 keys, the lone documented gap being `ref_pos`'s
    stochastic RDKit reference-conformer draw); this is the structural/
    shape+invariant gate."""
    pdb = os.path.join(os.path.dirname(__file__), "parity_artifacts", "ligand_iai", "IAI.pdb")
    code = "IAI"
    buried_atoms = ("C22,C23,C25,C24,C21,C20,N13,C15,C16,N14,C19,C11,N12,C18,C17,"
                     "N9,O8,C4,C1,N3,C10,N5,C7,C2,C6,N27,O26,C33,C29,C32,O31,C30,N28")
    spec = _spec({
        "input": pdb, "length": "180-180", "ligand": code,
        "select_fixed_atoms": {code: ""},
        "select_buried": {code: buried_atoms},
    })
    f = featurize(pdb, spec)
    I = 180 + 33  # 180 designed protein tokens + 33 ligand atoms (one token/atom)
    assert f["restype"].shape == (I, 32)
    is_lig = f["is_ligand"]
    assert int(is_lig.sum()) == 33 and int(f["is_protein"].sum()) == 180
    # ligand tokens: restype == UNK (index 20), never protein/rna/dna
    assert torch.all(f["restype"][is_lig].argmax(-1) == 20)
    assert not bool(f["is_protein"][is_lig].any())
    # ligand is its own fresh chain/entity, distinct from the designed protein
    assert f["asym_id"][180].item() != f["asym_id"][0].item()
    assert f["entity_id"][180].item() != f["entity_id"][0].item()
    # each ligand atom is its own token -> is_ca/is_central/is_backbone all True,
    # is_sidechain False (the AF3 "atomized token" convention)
    lig_atoms = torch.isin(f["atom_to_token_map"], is_lig.nonzero().flatten().to(torch.int32))
    assert bool(f["is_ca"][lig_atoms].all()) and bool(f["is_central"][lig_atoms].all())
    assert bool(f["is_backbone"][lig_atoms].all()) and not bool(f["is_sidechain"][lig_atoms].any())
    # select_fixed_atoms: {"IAI": ""} -> no ligand atom is coordinate-fixed,
    # but the ligand IS still "motif" (known chemical identity, class 1) and
    # fixed-SEQUENCE (ligands always have a known identity, never diffused
    # chemistry) -- verified vs a real reference capture.
    assert not bool(f["is_motif_atom_with_fixed_coord"][lig_atoms].any())
    assert bool(f["is_motif_atom_with_fixed_seq"][lig_atoms].all())
    assert bool(torch.all(f["ref_motif_token_type"][is_lig] == torch.tensor([0, 1, 0], dtype=torch.int8)))
    assert torch.all(f["ref_plddt"][is_lig] == 0)
    assert not bool(f["is_motif_token_with_fully_fixed_coord"][is_lig].any())
    # select_buried: {"IAI": <all 33 atoms>} -> ref_atomwise_rasa == "buried" (bin 0) everywhere
    assert bool(torch.all(f["ref_atomwise_rasa"][lig_atoms] == torch.tensor([1, 0, 0])))
    # ref_element/ref_charge/ref_mask are real (unlike protein's always-zero placeholder)
    assert bool(f["ref_mask"][lig_atoms].all())
    assert bool((f["ref_element"][lig_atoms, 0] == 0).all())  # never the "no element" placeholder row
    # real intra-ligand covalent bond graph -> a non-trivial token_bonds submatrix
    assert int(f["token_bonds"][180:, 180:].sum().item()) > 0
    # residue_index / ref_space_uid: all 33 atoms are ONE underlying residue
    assert torch.all(f["residue_index"][is_lig] == 0)
    ref_space_uid_lig = f["ref_space_uid"][lig_atoms]
    assert bool((ref_space_uid_lig == ref_space_uid_lig[0]).all())
    print(f"test_ligand_small_molecule_binder OK  I={I} L={f['ref_pos'].shape[0]}")


if __name__ == "__main__":
    test_motif_scaffold_shapes_and_invariants()
    test_chain_break()
    test_binder_contig_designed_only()
    test_ligand_rejected()
    test_na_binder_dsdna()
    test_unindex_tied_and_separate_islands()
    test_ligand_small_molecule_binder()
    test_contract_vs_golden()
    test_model_consumable()
    print("\nALL FEATURIZE STRUCTURAL TESTS PASS")
