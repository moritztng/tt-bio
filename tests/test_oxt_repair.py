"""Terminal OXT repair (tt_bio.oxt), the port of OpenDDE 6685cef, whose cases
(tests/test_structure_output.py there) these follow."""

import importlib
import os
import sys

import numpy as np
import pytest
import torch

from tt_bio.oxt import CCD_TERMINAL_REF, repair_terminal_oxt, repair_terminal_oxt_from_feats

ALA_NAMES = ["N", "CA", "C", "O", "CB", "OXT"]
# CCD ideal ALA, N/CB from tt_bio/data/protein_ref_conformers.json, C/CA/O/OXT from the table
ALA = np.array([[-0.966, 0.493, 1.5], [0.257, 0.418, 0.692], [-0.094, 0.017, -0.716],
                [-1.056, -0.682, -0.923], [1.204, -0.62, 1.296], [0.661, 0.439, -1.742]],
               dtype=np.float32)
ROT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
C, CA, O, OXT = 2, 1, 3, 5


def _residue(shift=(10.0, 20.0, 30.0)):
    return (ALA @ ROT.T + np.asarray(shift, dtype=np.float32)).astype(np.float32)


def _repair(coord, n_res=1, chains=None, mol=None, bonded=None):
    n = len(ALA_NAMES)
    names = ALA_NAMES * n_res
    chains = chains if chains is not None else ["A"] * (n * n_res)
    res_id = np.repeat(np.arange(1, n_res + 1), n)
    is_protein = np.ones(len(names), bool) if mol is None else mol
    return repair_terminal_oxt(coord, names, ["ALA"] * len(names), chains, res_id, is_protein,
                               bonded=bonded)


def _check_rebuilt(xyz):
    c, ca, o, oxt = xyz[C], xyz[CA], xyz[O], xyz[OXT]
    assert 1.0 <= np.linalg.norm(oxt - c) <= 1.7
    assert np.linalg.norm(oxt - o) >= 1.8
    normal = np.cross(ca - c, o - c)
    assert abs(np.dot(normal / np.linalg.norm(normal), oxt - c)) < 1e-2   # CA-C-O plane (CCD ideal is 3 mA off it)
    u, v = o - c, oxt - c
    ang = np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
    assert 115.0 < ang < 130.0


@pytest.mark.parametrize("bad", ["on_c", "far", "on_o", "nan"])
def test_invalid_oxt_is_rebuilt_to_the_ccd_position(bad):
    xyz = _residue()
    expect = xyz[OXT].copy()
    xyz[OXT] = {"on_c": xyz[C], "far": xyz[C] + [3.0, 0.0, 0.0], "on_o": xyz[O],
                "nan": [np.nan] * 3}[bad]
    before = xyz.copy()
    out, n = _repair(xyz)
    assert n == 1
    _check_rebuilt(out)
    np.testing.assert_allclose(out[OXT], expect, atol=1e-5)    # the rigid CCD placement
    np.testing.assert_array_equal(np.delete(out, OXT, 0), np.delete(before, OXT, 0))
    np.testing.assert_array_equal(xyz, before)                  # input not modified


def test_valid_oxt_is_bit_identical():
    xyz = _residue()
    xyz[OXT] += [0.05, -0.03, 0.02]
    out, n = _repair(xyz)
    assert n == 0 and out is not xyz
    np.testing.assert_array_equal(out, xyz)


def test_bad_anchor_is_skipped():
    xyz = _residue()
    xyz[CA] = xyz[C] + 2.5 * (xyz[CA] - xyz[C]) / np.linalg.norm(xyz[CA] - xyz[C])
    xyz[OXT] = xyz[C]
    out, n = _repair(xyz)
    assert n == 0
    np.testing.assert_array_equal(out, xyz)


@pytest.mark.parametrize("atom", [C, CA, O])
def test_nan_anchor_is_skipped(atom):
    xyz = _residue()
    xyz[atom] = np.nan
    xyz[OXT] = [100.0, 100.0, 100.0]
    out, n = _repair(xyz)
    assert n == 0
    np.testing.assert_array_equal(out, xyz)


def test_externally_bonded_carboxylate_is_skipped():
    for atom in (C, O, OXT):
        xyz = _residue()
        xyz[OXT] = xyz[C]
        bonded = np.zeros(len(ALA_NAMES), bool); bonded[atom] = True
        out, n = _repair(xyz, bonded=bonded)
        assert n == 0
        np.testing.assert_array_equal(out, xyz)


def test_only_the_terminal_residue_and_proteins_are_touched():
    r1, r2 = _residue((0, 0, 0)), _residue((3.8, 0, 0))
    r1[OXT] = r1[C]                       # an OXT on a non-terminal residue
    r2[OXT] = r2[C]
    xyz = np.concatenate([r1, r2])
    out, n = _repair(xyz, n_res=2)
    assert n == 1
    np.testing.assert_array_equal(out[:6], xyz[:6])
    _check_rebuilt(out[6:])
    # the same atoms flagged ligand / nucleic acid (not protein) are left alone
    out, n = _repair(xyz, n_res=2, mol=np.zeros(12, bool))
    assert n == 0
    np.testing.assert_array_equal(out, xyz)


def test_two_chains_independently():
    a, b = _residue((0, 0, 0)), _residue((30, 0, 0))
    a[OXT] = [100.0, 100.0, 100.0]        # A bad, B already valid
    xyz = np.concatenate([a, b])
    out, n = repair_terminal_oxt(xyz, ALA_NAMES * 2, ["ALA"] * 12, ["A"] * 6 + ["B"] * 6,
                                 [1] * 12, np.ones(12, bool))
    assert n == 1
    _check_rebuilt(out[:6])
    np.testing.assert_array_equal(out[6:], xyz[6:])
    a2 = a.copy(); b2 = b.copy(); b2[OXT] = b2[O]
    out, n = repair_terminal_oxt(np.concatenate([a2, b2]), ALA_NAMES * 2, ["ALA"] * 12,
                                 ["A"] * 6 + ["B"] * 6, [1] * 12, np.ones(12, bool))
    assert n == 2
    _check_rebuilt(out[:6]); _check_rebuilt(out[6:])


def test_every_ccd_reference_passes_its_own_gates():
    for res, ref in CCD_TERMINAL_REF.items():
        c, ca, o, oxt = ref
        assert 1.0 <= np.linalg.norm(oxt - c) <= 1.7, res
        assert np.linalg.norm(oxt - o) >= 1.8, res
        assert 1.3 <= np.linalg.norm(ca - c) <= 1.7, res


def _peptide_feats(seq="GAWK", **kw):
    from tt_bio.protenix_data import build_complex_features
    return build_complex_features([(seq, None)], **kw)


def _names_and_res(feats):
    from tt_bio.protenix_data import restype_to_resname
    a2t = feats["atom_to_token_idx"].tolist()
    names = ["".join(chr(c + 32) for c in ch).strip()
             for ch in feats["ref_atom_name_chars"].argmax(-1).tolist()]
    rn = restype_to_resname(feats["restype"].argmax(-1))
    return np.array(names), np.array([rn[t] for t in a2t])


def _ideal_chain(feats, names):
    """A chain of ideal residues 3.8 A apart, laid out from the CCD conformers."""
    from tt_bio.protenix_data import load_ref_conformers, residue_atoms, restype_to_resname
    conf = load_ref_conformers()
    rn = restype_to_resname(feats["restype"].argmax(-1))
    a2t = feats["atom_to_token_idx"].tolist()
    xyz = np.zeros((len(a2t), 3), np.float32)
    for i, t in enumerate(a2t):
        atoms = residue_atoms(rn[t])
        if names[i] == "OXT":
            xyz[i] = CCD_TERMINAL_REF[rn[t]][3] - CCD_TERMINAL_REF[rn[t]][0] \
                + conf[rn[t]][atoms.index("C")].numpy()
        else:
            xyz[i] = conf[rn[t]][atoms.index(names[i])].numpy()
        xyz[i] += [3.8 * t, 0.0, 0.0]
    return xyz


def test_feats_wrapper_finds_the_terminal_oxt():
    feats = _peptide_feats()
    names, res = _names_and_res(feats)
    xyz = _ideal_chain(feats, names)
    i_oxt = int(np.flatnonzero(names == "OXT")[0])
    i_c = [i for i in range(len(names)) if names[i] == "C"][-1]
    want = xyz[i_oxt].copy()
    xyz[i_oxt] = xyz[i_c]
    out, n = repair_terminal_oxt_from_feats(torch.from_numpy(xyz), feats, names, res)
    assert n == 1
    np.testing.assert_allclose(out[i_oxt], want, atol=1e-5)
    # a covalent bond on the last residue (here to a ligand) disables the repair
    feats["token_bonds"][-1, 0] = feats["token_bonds"][0, -1] = 1.0
    out, n = repair_terminal_oxt_from_feats(torch.from_numpy(xyz), feats, names, res)
    assert n == 0
    np.testing.assert_array_equal(out, xyz)


@pytest.mark.parametrize("fmt", ["cif", "pdb"])
def test_written_file_carries_the_rebuilt_oxt(tmp_path, fmt):
    import biotite.structure.io as strucio
    from tt_bio.main import _write_protenix_structure

    feats = _peptide_feats()
    names, _ = _names_and_res(feats)
    xyz = _ideal_chain(feats, names)
    i_oxt = int(np.flatnonzero(names == "OXT")[0])
    i_c = [i for i in range(len(names)) if names[i] == "C"][-1]
    want = xyz[i_oxt].copy()
    xyz[i_oxt] = xyz[i_c]
    coords = torch.from_numpy(xyz.copy())
    before = coords.clone()
    out = tmp_path / f"model.{fmt}"
    _write_protenix_structure(coords, feats, None, out, fmt)
    torch.testing.assert_close(coords, before, rtol=0, atol=0)
    arr = strucio.load_structure(str(out), model=1)
    oxt = arr[arr.atom_name == "OXT"]
    assert len(oxt) == 1
    np.testing.assert_allclose(oxt.coord[0], want, atol=2e-3)
    rest = arr[arr.atom_name != "OXT"].coord
    np.testing.assert_allclose(rest, np.delete(xyz, i_oxt, 0), atol=2e-3)


def _upstream():
    src = os.environ.get("OPENDDE_SRC", "/tmp/tfgsrc/up")
    if not os.path.isdir(os.path.join(src, "opendde")):
        pytest.skip("OpenDDE source not available")
    sys.path.insert(0, src)
    try:
        return importlib.import_module("opendde.data.utils")
    except Exception as e:                                   # pragma: no cover
        pytest.skip(f"OpenDDE import failed: {e}")
    finally:
        sys.path.remove(src)


def test_matches_upstream_on_the_same_reference():
    """Same inputs, ref_pos = the CCD ideal table: the rebuilt OXT agrees with OpenDDE's."""
    from biotite.structure import AtomArray
    up = _upstream()
    rng = np.random.default_rng(0)
    worst = 0.0
    for res, ref in CCD_TERMINAL_REF.items():
        for trial in range(5):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            q = q * np.sign(np.linalg.det(q))
            pred = (ref @ q.T + rng.normal(scale=5.0, size=3)).astype(np.float32)
            pred[:3] += rng.normal(scale=0.03, size=(3, 3)).astype(np.float32)
            pred[3] = [pred[0], pred[2], [np.nan] * 3, pred[0] + 3.0, pred[0] + 0.2][trial]
            arr = AtomArray(4)
            arr.atom_name[:] = ["C", "CA", "O", "OXT"]
            arr.res_name[:] = res; arr.chain_id[:] = "A"; arr.res_id[:] = 1
            arr.coord = pred.copy()
            arr.set_annotation("mol_type", np.array(["protein"] * 4))
            arr.set_annotation("label_asym_id", np.array(["A"] * 4))
            arr.set_annotation("ref_pos", ref.copy())
            arr.set_annotation("ref_mask", np.ones(4, dtype=int))
            n_up = up._repair_terminal_oxt_coordinates(arr)
            ours, n = repair_terminal_oxt(pred, ["C", "CA", "O", "OXT"], [res] * 4, ["A"] * 4,
                                          [1] * 4, np.ones(4, bool))
            assert n == n_up == 1
            worst = max(worst, float(np.abs(ours - arr.coord).max()))
    print(f"max |ours - upstream| = {worst:.3g} A")
    assert worst == 0.0
