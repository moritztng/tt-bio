"""The `mol_type` -> entity-flag adapter, and the two conventions it has to tell apart.

Host-only: numpy, no ttnn, no device.

D16. Upstream weights dna and rna at 5.0 and ligands at 10.0 in the diffusion MSE, and
`losses.mse` reaches that weighting only through `is_dna` / `is_rna` / `is_ligand`. Only
OpenFold3's vendored featuriser emits those three names. Protenix-v2, Boltz-2 and BoltzGen
emit the same fact as a single integer `mol_type` column, so before `entity_flags` existed
every nucleic-acid and ligand token those three trained on was weighted as protein.

WHAT IS LOAD-BEARING HERE. The table in `objectives.MOL_TYPE_CONVENTIONS` is a claim about
three other stacks' data, and a table copied out of someone else's module is exactly the kind
that goes stale without anything saying so. So the first test below does not pin the table --
it RE-DERIVES every entry from the live source and compares. If Boltz reorders `chain_types`
or Protenix renumbers `MOL_TYPE_IDS`, this fails here instead of silently retraining nucleic
acids at the wrong weight.

The second load-bearing test is the invariance the unresolved dna/rna split rests on. The two
conventions swap dna and rna, an integer column cannot say which is which, and the adapter
proceeds anyway -- which is only safe because upstream weights the two identically. That is
asserted as an equality of numbers AND broken on purpose, so it is a checked invariant rather
than a comment.
"""
from __future__ import annotations

import numpy as np
import pytest

from tt_bio.train import losses, objectives


def test_the_convention_table_matches_each_stacks_live_source():
    """Re-derive all three stacks' integer domain from their own modules."""
    af3 = objectives.MOL_TYPE_CONVENTIONS["af3"]
    boltz = objectives.MOL_TYPE_CONVENTIONS["boltz"]

    # Boltz-2 and BoltzGen both import this one module; neither ships its own.
    from tt_bio.data import const
    assert const.chain_type_ids == {"PROTEIN": boltz["protein"], "DNA": boltz["dna"],
                                    "RNA": boltz["rna"], "NONPOLYMER": boltz["ligand"]}

    # Protenix-v2's own featuriser enum.
    from tt_bio.protenix_data import MOL_TYPE_IDS
    assert MOL_TYPE_IDS == {"protein": af3["protein"], "rna": af3["rna"],
                            "dna": af3["dna"], "ligand": af3["ligand"]}

    # OpenFold3's own enum, which is where the `af3` row's authority actually comes from:
    # Protenix agreeing with it is a second reading, not the source.
    of3 = pytest.importorskip(
        "tt_bio._vendor.openfold3.core.data.resources.residues",
        reason="the vendored OpenFold3 tree is not on this checkout").MoleculeType
    assert (int(of3.PROTEIN), int(of3.RNA), int(of3.DNA), int(of3.LIGAND)) == (
        af3["protein"], af3["rna"], af3["dna"], af3["ligand"])

    # And the finding that makes one shared table necessary rather than convenient: the two
    # conventions agree on protein and ligand and SWAP dna and rna.
    assert af3["protein"] == boltz["protein"] and af3["ligand"] == boltz["ligand"]
    assert af3["dna"] == boltz["rna"] and af3["rna"] == boltz["dna"]


def _batch(mol_type, seed=0):
    rng = np.random.default_rng(seed)
    n = len(mol_type)
    true_xyz = rng.standard_normal((n, 3)) * 5.0
    return ({"true_xyz": true_xyz, "coord_mask": np.ones(n),
             "mol_type": np.asarray(mol_type)},
            {"pred_xyz": true_xyz + rng.standard_normal((n, 3))})


def test_ligand_weighting_reaches_losses_mse_from_a_mol_type_column():
    """The defect itself: a batch with only `mol_type` must still be weighted."""
    batch, outputs = _batch([0, 0, 0, 0, 3, 3])
    total, breakdown, seeds = objectives.af3_loss(batch, outputs, {"mse": 4.0})
    assert breakdown["mse"].get("without") is None
    assert breakdown["mse"]["derived"]["counts"] == {"is_dna": 0, "is_rna": 0, "is_ligand": 2}

    bare = {k: v for k, v in batch.items() if k != "mol_type"}
    _, b_bare, s_bare = objectives.af3_loss(bare, outputs, {"mse": 4.0})
    assert b_bare["mse"]["without"] == ["is_dna", "is_rna", "is_ligand"]
    # The weighting is worth something, which is the whole point of deriving it.
    assert breakdown["mse"]["value"] != b_bare["mse"]["value"]
    assert not np.allclose(seeds["pred_xyz"], s_bare["pred_xyz"])


def test_the_dna_rna_swap_cannot_move_a_number_while_upstream_weights_them_equally():
    """The invariance the unresolved split rests on, as an equality and then broken."""
    mt = [0, 1, 2, 3, 1, 2]
    batch, outputs = _batch(mt)
    v = {}
    for conv in ("af3", "boltz"):
        _, b, s = objectives.af3_loss({**batch, objectives.CONVENTION_KEY: conv},
                                      outputs, {"mse": 4.0})
        v[conv] = (b["mse"]["value"], np.asarray(s["pred_xyz"]))
    assert v["af3"][0] == v["boltz"][0]
    assert np.array_equal(v["af3"][1], v["boltz"][1])   # bit-identical, not merely close

    # Break it: the flags themselves DO differ, so the invariance is a property of the equal
    # weights and not of the adapter returning the same thing twice.
    f_af3 = objectives.entity_flags(mt, "af3")["flags"]
    f_bz = objectives.entity_flags(mt, "boltz")["flags"]
    assert not np.array_equal(f_af3["is_dna"], f_bz["is_dna"])
    assert np.array_equal(f_af3["is_ligand"], f_bz["is_ligand"])
    # And with the weights separated, the same two conventions disagree.
    lo = losses.mse(outputs["pred_xyz"], batch["true_xyz"], batch["coord_mask"],
                    w_dna=5.0, w_rna=1.0, **f_af3)[0]
    hi = losses.mse(outputs["pred_xyz"], batch["true_xyz"], batch["coord_mask"],
                    w_dna=5.0, w_rna=1.0, **f_bz)[0]
    assert lo != hi


def test_an_unnamed_convention_is_refused_once_the_nucleic_weights_differ(monkeypatch):
    """The guard, executed. A comment saying "safe because 5.0 == 5.0" is not a guard."""
    import inspect
    real = losses.mse
    sig = inspect.signature(real)
    params = dict(sig.parameters)
    params["w_rna"] = params["w_rna"].replace(default=1.0)

    def skewed(*a, **kw):
        return real(*a, **kw)
    skewed.__signature__ = sig.replace(parameters=list(params.values()))
    monkeypatch.setattr(losses, "mse", skewed)

    with pytest.raises(ValueError, match="has become load-bearing"):
        objectives.entity_flags([0, 1, 2, 3])
    # Naming the convention is the way through, and it still works.
    assert objectives.entity_flags([0, 1, 2, 3], "boltz")["resolved"] is True


def test_a_class_the_convention_does_not_define_is_refused_not_treated_as_protein():
    with pytest.raises(ValueError, match=r"mol_type carries \[7\]"):
        objectives.entity_flags([0, 1, 7])
    # A one-hot or a float column compares unequal to every id and would derive three
    # all-zero flags, which is bit-identical to deriving nothing at all.
    with pytest.raises(TypeError, match="integer class column"):
        objectives.entity_flags(np.zeros(4, dtype=np.float32))


def test_native_flags_win_and_the_batch_is_not_mutated():
    """A featuriser that emits the three names is untouched, and nothing is edited in place."""
    batch, outputs = _batch([0, 0, 3, 3])
    native = {**batch, "is_dna": np.zeros(4), "is_rna": np.zeros(4),
              "is_ligand": np.array([0.0, 0.0, 1.0, 1.0])}
    before = set(native)
    _, b, _ = objectives.af3_loss(native, outputs, {"mse": 4.0})
    assert "derived" not in b["mse"] and b["mse"].get("without") is None
    assert set(native) == before

    keys_before = set(batch)
    objectives.af3_loss(batch, outputs, {"mse": 4.0})
    assert set(batch) == keys_before, "af3_loss edited the caller's batch"
