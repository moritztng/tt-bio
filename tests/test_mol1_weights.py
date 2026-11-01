"""Mol-1's trunk layout read off its weights: PairMixer blocks and pair-width adapters.

Card-free. The on-device fold is perf/mol1_smoke (0.68 A CA RMSD to 1UBQ on Blackhole).
"""
import torch

from tt_bio.protenix import trunk_recycles
from tt_bio.protenix_weights import remap_pair_adapters
from tt_bio.weights import ARTIFACTS, split_hf


def test_adapters_keyed_by_the_block_they_precede():
    sd = {"pairformer_stack.adapters.before_block_8.pair_adapter.weight": torch.zeros(320, 448),
          "pairformer_stack.adapters.after_blocks.pair_adapter.weight": torch.ones(448, 320),
          "pairformer_stack.blocks.0.tri_mul_out.linear_z.weight": torch.zeros(448, 448)}
    out = remap_pair_adapters(sd, "pairformer_stack.", 48)
    assert set(out) == {"adapters.8.weight", "adapters.48.weight"}
    assert out["adapters.8.weight"].shape == (320, 448)
    assert out["adapters.48.weight"].shape == (448, 320)


def test_single_width_checkpoint_has_no_adapters():
    sd = {"pairformer_stack.blocks.0.tri_att_start.linear.weight": torch.zeros(8, 256),
          "confidence_head.pairformer_stack.adapters.after_blocks.pair_adapter.weight": torch.zeros(4, 4)}
    assert remap_pair_adapters(sd, "pairformer_stack.", 48) == {}


def test_recycles_read_off_the_checkpoint():
    v2 = {"template_embedder.pairformer_stack.blocks.0.tri_mul_out.linear_z.weight": torch.zeros(1)}
    mol1 = {**v2, "pairformer_stack.adapters.before_block_8.pair_adapter.weight": torch.zeros(1)}
    assert trunk_recycles(v2) == 10
    assert trunk_recycles(mol1) == 20


def test_checkpoint_source_in_a_hub_subdirectory():
    art = ARTIFACTS["mol1"]
    assert split_hf(art.sources[0]) == ("RUIMA294264/Mol-1", "checkpoint/mol-1-base.pt")
    assert art.sources[1].startswith("https://mol-1-release.oss-cn-shanghai.aliyuncs.com/")
    assert split_hf("hf://moritztng/boltz-2/boltz2_conf.ckpt") == ("moritztng/boltz-2", "boltz2_conf.ckpt")


def _ccd_pickle(path, mols):
    import pickle
    path.write_bytes(pickle.dumps(mols))
    return path


def test_conformer_table_takes_each_codes_ref_conformer(tmp_path):
    """Upstream reads conformer mol.ref_conf_id with names from mol.atom_map; so must the table."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from tt_bio.protenix_data import load_conformer_table, write_conformer_table
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMultipleConfs(mol, 3, randomSeed=0)
    mol.ref_conf_id = 2
    mol.atom_map = {f"A{i}": i for i in range(mol.GetNumAtoms())}
    out = tmp_path / "t.npz"
    assert write_conformer_table(_ccd_pickle(tmp_path / "c.pkl", {"MOH": mol, "NIL": Chem.Mol()}), out) == 1
    table = load_conformer_table(out)
    assert set(table) == {"MOH"}
    want = mol.GetConformer(2).GetPositions().astype(np.float32)
    assert np.array_equal(np.stack([table["MOH"][f"A{i}"] for i in range(mol.GetNumAtoms())]), want)


def test_conformer_pickle_may_not_construct_anything_else(tmp_path):
    import os
    import pickle
    import pytest
    from tt_bio.protenix_data import write_conformer_table

    class Boom:
        def __reduce__(self):
            return (os.getcwd, ())
    with pytest.raises(pickle.UnpicklingError, match="posix.getcwd|os.getcwd"):
        write_conformer_table(_ccd_pickle(tmp_path / "bad.pkl", {"X": Boom()}), tmp_path / "t.npz")
    assert not (tmp_path / "t.npz").exists()


def test_mol1_fetches_its_own_conformers():
    from tt_bio.weights import artifacts_for
    arts = {a.key: a for a in artifacts_for("mol1")}
    assert set(arts) == {"mol1", "mol1-ccd", "mols"}
    ccd = arts["mol1-ccd"]
    assert split_hf(ccd.sources[0]) == ("RUIMA294264/Mol-1", "common/components.cif.rdkit_mol.pkl")
    assert ccd.derived.producer == "rdkit-conformers" and ccd.derived.expect == ("ccd_table.npz",)
    assert not any(a.derived and a.derived.producer == "rdkit-conformers" for a in artifacts_for("protenix-v2"))


def test_only_mol1_folds_read_their_own_conformers(tmp_path, monkeypatch):
    from tt_bio import weights
    from tt_bio.worker import _ensure_local_artifacts
    monkeypatch.setenv("TT_BIO_CACHE", str(tmp_path))
    monkeypatch.setattr(weights, "fetch", lambda key, **k: tmp_path / key)
    cfgs = {m: {"model": m, "msa_dir": None} for m in ("mol1", "protenix-v2")}
    for cfg in cfgs.values():
        _ensure_local_artifacts(cfg)
    assert cfgs["mol1"]["conformer_table"] == str(tmp_path / "mol1-ccd" / "ccd_table.npz")
    assert "conformer_table" not in cfgs["protenix-v2"]
