"""Host-only contract tests for the --model openfold3 CLI wiring (P13/S6).

Pins: the predict --model choice accepts openfold3; openfold3 is treated as an
MSA-dependent model by _resolve_msa_default (never silently single-sequence unless
explicitly asked); the worker resolves the OF3 checkpoint via $OF3_CKPT or the
cache and fails with a clear message otherwise. No device, no network.
"""
from __future__ import annotations

import pytest


def test_model_choice_accepts_openfold3():
    from tt_bio.main import predict

    model_opt = next(p for p in predict.params if p.name == "model")
    assert "openfold3" in model_opt.type.choices


def test_openfold3_is_msa_dependent(tmp_path):
    """No explicit source + no local DB -> falls back to the online server (True),
    exactly like boltz2/protenix-v2; single_sequence + explicit source is rejected."""
    import click

    from tt_bio.main import _resolve_msa_default

    use_server, db = _resolve_msa_default(
        "openfold3", False, None, None, False, tmp_path, None, "http://msa.example")
    assert use_server is True and db is None

    use_server, db = _resolve_msa_default(
        "openfold3", False, "/some/db", None, False, tmp_path, None, "http://msa.example")
    assert (use_server, db) == (False, "/some/db")

    with pytest.raises(click.BadParameter):
        _resolve_msa_default(
            "openfold3", True, "/some/db", None, True, tmp_path, None, "http://msa.example")


def test_worker_resolves_of3_checkpoint(tmp_path, monkeypatch):
    from tt_bio.worker import _ensure_local_artifacts

    # A real torch archive, not arbitrary bytes: the worker now verifies the file it is
    # handed instead of only checking that the path exists, so a truncated manual copy
    # is reported by name rather than dying later inside torch.load.
    import torch

    ckpt = tmp_path / "of3-p2-155k.pt"
    torch.save({"stub": torch.zeros(1)}, ckpt)
    monkeypatch.setenv("OF3_CKPT", str(ckpt))
    monkeypatch.setenv("BOLTZ_CACHE", str(tmp_path))
    cfg = {"model": "openfold3", "msa_dir": None}
    _ensure_local_artifacts(cfg)
    assert cfg["of3_ckpt"] == str(ckpt)
    assert cfg["msa_dir"]  # resolved to a writable dir


def test_worker_errors_clearly_without_checkpoint(tmp_path, monkeypatch):
    from tt_bio.worker import _ensure_local_artifacts

    monkeypatch.delenv("OF3_CKPT", raising=False)
    monkeypatch.setenv("BOLTZ_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="OF3_CKPT"):
        _ensure_local_artifacts({"model": "openfold3", "msa_dir": None})


def _yaml(tmp_path, body):
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return p


def test_template_map_reads_per_chain_npz(tmp_path):
    from tt_bio.worker import _openfold3_template_map

    npz = tmp_path / "tmpl.npz"
    npz.write_bytes(b"stub")
    p = _yaml(tmp_path, f"""version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: MKVL
      templates: {npz}
  - protein:
      id: C
      sequence: ACGT
""")
    assert _openfold3_template_map(p) == {"A": str(npz), "B": str(npz)}


def test_template_map_rejects_missing_file(tmp_path):
    from tt_bio.worker import _openfold3_template_map

    p = _yaml(tmp_path, """version: 1
sequences:
  - protein:
      id: A
      sequence: MKVL
      templates: /nonexistent/tmpl.npz
""")
    with pytest.raises(RuntimeError, match="does not exist"):
        _openfold3_template_map(p)


def test_template_map_rejects_non_protein_chain(tmp_path):
    from tt_bio.worker import _openfold3_template_map

    npz = tmp_path / "tmpl.npz"
    npz.write_bytes(b"stub")
    p = _yaml(tmp_path, f"""version: 1
sequences:
  - rna:
      id: R
      sequence: ACGU
      templates: {npz}
""")
    with pytest.raises(RuntimeError, match="only valid on protein chains"):
        _openfold3_template_map(p)


def test_template_map_ignores_fasta_and_template_free_yaml(tmp_path):
    from tt_bio.worker import _openfold3_template_map

    fa = tmp_path / "in.fasta"
    fa.write_text(">A|protein\nMKVL\n")
    assert _openfold3_template_map(fa) == {}
    p = _yaml(tmp_path, "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKVL\n")
    assert _openfold3_template_map(p) == {}


def test_of3_constraints_reject_covalent_bonds(tmp_path):
    import yaml

    from tt_bio.worker import _refuse_dropped_bonds

    p = tmp_path / "covalent.yaml"
    p.write_text(yaml.safe_dump({
        "version": 1,
        "sequences": [{"protein": {"id": "A", "sequence": "GACGAC"}}],
        "constraints": [{"bond": {"atom1": ["A", 1, "SG"], "atom2": ["A", 4, "SG"]}}],
    }))
    try:
        _refuse_dropped_bonds(p)
    except RuntimeError as e:
        assert "covalent bonds" in str(e)
    else:
        raise AssertionError("covalent constraint must raise, not be ignored")

    p2 = tmp_path / "plain.yaml"
    p2.write_text(yaml.safe_dump({
        "version": 1,
        "sequences": [{"protein": {"id": "A", "sequence": "GACGAC"}}],
    }))
    _refuse_dropped_bonds(p2)  # no constraints: passes


def test_of3_chains_reject_ligands_blank_and_empty():
    from tt_bio.worker import _validate_openfold3_chains

    with pytest.raises(RuntimeError, match="no protein/nucleic-acid"):
        _validate_openfold3_chains([])
    with pytest.raises(RuntimeError, match="polymer-only"):
        _validate_openfold3_chains([("L", "CCD_ATP", None, "ligand")])
    with pytest.raises(RuntimeError, match="empty/whitespace-only"):
        _validate_openfold3_chains([("A", "   ", None, "protein")])
    with pytest.raises(RuntimeError, match="empty/whitespace-only"):
        _validate_openfold3_chains([("A", "MKVL", None, "protein"), ("B", "", None, "rna")])
    # valid polymer chains pass; unknown residue codes are upstream-compatible (UNK warning)
    _validate_openfold3_chains([("A", "MKVLXXX", None, "protein"), ("R", "ACGU", None, "rna")])


def _of3_query(chains):
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    iqs = InferenceQuerySet.model_validate({"queries": {"q": {
        "query_name": "q", "use_msas": True, "use_paired_msas": False,
        "use_main_msas": True, "covalent_bonds": None, "chains": chains}}})
    return next(iter(iqs.queries.values()))


def _chain(cid, seq, mtype, msa=None):
    return {"molecule_type": mtype, "chain_ids": [cid], "sequence": seq,
            "non_canonical_residues": None, "smiles": None, "ccd_codes": None,
            "paired_msa_file_paths": None,
            "main_msa_file_paths": [str(msa)] if msa else None,
            "template_alignment_file_path": None, "template_entry_chain_ids": None,
            "sdf_file_path": None}


def test_single_sequence_augment_writes_upstream_one_row_a3m(tmp_path):
    """--single_sequence = upstream's no-MSA mode: every MSA-less protein/RNA
    chain gets a one-row a3m holding exactly its sequence (upstream's bytes,
    no trailing newline) and the MSA stack stays on."""
    from tt_bio.openfold3_data import augment_openfold3_msas_with_query_sequence

    q = _of3_query([_chain("A", "MKVL", "PROTEIN"), _chain("R", "ACGU", "RNA")])
    q = augment_openfold3_msas_with_query_sequence(q, tmp_path)
    for chain, seq in zip(q.chains, ("MKVL", "ACGU")):
        (path,) = chain.main_msa_file_paths
        assert path.name == "colabfold_main.a3m"  # canonical basename OF3 filters on
        assert path.read_bytes() == b">query\n" + seq.encode()
        # the dummy must not sit in the shared hash cache: a one-row alignment
        # is not a real MSA and must never satisfy a later run's cache lookup
        assert path.parent.parent.name == "dummy"
    assert q.use_msas and q.use_main_msas

    # idempotent + content-addressed: a second call reuses the same file
    again = augment_openfold3_msas_with_query_sequence(q, tmp_path)
    assert [str(p) for c in again.chains for p in c.main_msa_file_paths] == [
        str(p) for c in q.chains for p in c.main_msa_file_paths]


def test_single_sequence_augment_preserves_user_and_cached_msas(tmp_path):
    """Chains that already carry an alignment keep it (upstream only fills
    chains whose main_msa_file_paths are unset); DNA gets no dummy."""
    from tt_bio.openfold3_data import augment_openfold3_msas_with_query_sequence

    user_a3m = tmp_path / "real.a3m"
    user_a3m.write_text(">1\nMKVL\n")
    q = _of3_query([_chain("A", "MKVL", "PROTEIN", msa=user_a3m),
                    _chain("B", "GACG", "PROTEIN"),
                    _chain("D", "ACGT", "DNA")])
    q = augment_openfold3_msas_with_query_sequence(q, tmp_path)
    a, b, d = q.chains
    assert [str(p) for p in a.main_msa_file_paths] == [str(user_a3m)]
    assert b.main_msa_file_paths[0].read_text() == ">query\nGACG"
    assert d.main_msa_file_paths is None
