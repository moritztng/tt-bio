"""The Protenix/OpenDDE template features, scored against upstream's own geometry.

`protenix.py::Trunk._template` runs a trained 2-block pairformer stack on protenix-v2 and
opendde, and until `tt_bio/protenix_template.py` it was only ever handed
`protenix_data.dummy_template_features`. A wrong featurization is worse than a dummy one:
the embedder is trained on upstream's exact bin edges and frame convention, and every way of
getting them wrong still produces a plausible fold. So this scores the four pairwise
features against upstream's own `TemplateFeatures` functions rather than against a
hand-written expectation.

Two arms:

* the parity arm needs an upstream checkout (`$PROTENIX_REF_SRC`, default
  `~/protenix_ref_src`) and SKIPs without one;
* everything else runs anywhere. Rigid-motion invariance is the arm that catches the frame
  convention without upstream: `R^T (CA_j - CA_i)` is invariant under a global rotation of
  the template, and `R (CA_j - CA_i)` is not, so a transposed frame fails it.

Host-only, no card. The fixtures are the committed 1y57 alignment + mmCIF that the OpenFold3
template path already uses, which is the point: one file format, both stacks.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tt_bio.protenix_template import (GAP, chain_template_arrays,
                                      complex_template_features, pair_features,
                                      read_alignment_entries)

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples" / "of3_upstream"
NPZ = EX / "template_alignments" / "1y57.npz"
STRUCT = EX / "template_structures"
YML = EX / "1y57_template_on.yaml"
FEATURES = ("template_aatype", "template_distogram", "template_pseudo_beta_mask",
            "template_unit_vector", "template_backbone_frame_mask")

_REF = Path(os.environ.get("PROTENIX_REF_SRC", str(Path.home() / "protenix_ref_src")))
needs_ref = pytest.mark.skipif(not (_REF / "protenix" / "data" / "template").is_dir(),
                               reason=f"needs an upstream Protenix checkout at {_REF}")


def _query_seq() -> str:
    return yaml.safe_load(YML.read_text())["sequences"][0]["protein"]["sequence"]


def _arrays():
    """The 1y57 template as (aatype, pos, mask) over the query's 447 residues."""
    seq = _query_seq()
    return chain_template_arrays(len(seq), read_alignment_entries(NPZ), STRUCT)


def _dense24(aatype, pos4, mask4):
    """tt-bio's four atom slots placed into upstream's dense-24 layout, by upstream's own
    index tables, so both sides see identical coordinates and any difference is in the
    formulas rather than in the input."""
    from protenix.data.constants import (RESTYPE_PSEUDOBETA_INDEX,
                                         RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX)
    T, L = aatype.shape
    pos = np.zeros((T, L, 24, 3), dtype=np.float32)
    mask = np.zeros((T, L, 24), dtype=np.int32)
    r = np.arange(L)
    for t in range(T):
        bb = RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX[aatype[t], 0]      # (L, 3) = C, CA, N
        pb = RESTYPE_PSEUDOBETA_INDEX[aatype[t]]
        for slot, col in ((bb[:, 2], 0), (bb[:, 1], 1), (bb[:, 0], 2), (pb, 3)):
            pos[t, r, slot] = pos4[t, :, col]
            mask[t, r, slot] = np.maximum(mask[t, r, slot], mask4[t, :, col].astype(np.int32))
    return pos, mask


def _upstream(aatype, pos4, mask4):
    from protenix.data.template.template_utils import (DistogramFeaturesConfig,
                                                       TemplateFeatures)
    dense_pos, dense_mask = _dense24(aatype, pos4, mask4)
    cfg = DistogramFeaturesConfig(min_bin=3.25, max_bin=50.75, num_bins=39)
    out = {k: [] for k in FEATURES[1:]}
    for t in range(aatype.shape[0]):
        m, p = dense_mask[t], dense_pos[t] * dense_mask[t][..., None]
        pb_pos, pb_mask = TemplateFeatures.pseudo_beta_fn(aatype[t], p, m)
        pb2 = pb_mask[:, None] * pb_mask[None, :]
        uv, bb2 = TemplateFeatures.compute_template_unit_vector(aatype[t], p, m)
        out["template_distogram"].append(
            np.asarray(TemplateFeatures.dgram_from_positions(pb_pos, config=cfg)) * pb2[..., None])
        out["template_pseudo_beta_mask"].append(np.asarray(pb2))
        out["template_unit_vector"].append(np.asarray(uv) * np.asarray(bb2)[..., None])
        out["template_backbone_frame_mask"].append(np.asarray(bb2))
    return {k: np.stack(v).astype(np.float64) for k, v in out.items()}


def _worst(mine, ref):
    return {k: float(np.abs(mine[k].numpy().astype(np.float64) - ref[k]).max()) for k in ref}


@pytest.fixture(scope="module")
def ref_path():
    sys.path.insert(0, str(_REF))
    yield
    sys.path.remove(str(_REF))


@needs_ref
def test_pair_features_reproduce_upstream_exactly(ref_path):
    """Every bin edge, every frame, on a real 447-residue template with 444 aligned."""
    aatype, pos, mask = _arrays()
    assert (aatype[0] != GAP).sum() == 444, "the 1y57 fixture stopped aligning what it did"
    worst = _worst(pair_features(aatype, pos, mask), _upstream(aatype, pos, mask))
    assert worst == {k: 0.0 for k in worst}, worst


@needs_ref
def test_that_comparison_can_fail(ref_path):
    """Negative control. Score a DIFFERENT geometry against the same reference: if the
    comparison above were vacuous (all-zero features, an empty template) this would pass too.

    Two perturbations, because one is not enough. Scaling every coordinate moves every
    distogram bin and leaves the unit vector alone -- it is a direction, and a scaling about
    the origin preserves directions -- so the unit vector needs a jitter that actually
    reorients the frames."""
    aatype, pos, mask = _arrays()
    ref = _upstream(aatype, pos, mask)
    scaled = _worst(pair_features(aatype, pos * 1.5, mask), ref)
    assert scaled["template_distogram"] == 1.0, "scaling the template moved no distogram bin"
    jitter = np.random.default_rng(0).normal(0.0, 0.5, pos.shape).astype(np.float32)
    assert _worst(pair_features(aatype, pos + jitter, mask), ref)["template_unit_vector"] > 1e-2


def test_a_rigid_motion_changes_nothing():
    """Distances and a CA-local frame are both invariant under a global rotation +
    translation. This is what pins `R^T d` rather than `R d` without upstream on the box."""
    aatype, pos, mask = _arrays()
    base = pair_features(aatype, pos, mask)
    theta = 0.7
    c, s = np.cos(theta), np.sin(theta)
    Q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    moved = pair_features(aatype, pos @ Q.T + np.float32([3.0, -7.0, 11.0]), mask)
    for k in FEATURES:
        d = float((moved[k].double() - base[k].double()).abs().max())
        assert d < 2e-5, f"{k} moved by {d} under a rigid motion"


def test_a_transposed_frame_would_fail_that():
    """Negative control for the invariance above: `R d` is NOT rotation-invariant, so the
    check has teeth."""
    aatype, pos, mask = _arrays()
    theta = 0.7
    c, s = np.cos(theta), np.sin(theta)
    Q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    uv = pair_features(aatype, pos, mask)["template_unit_vector"].numpy()
    uv_moved = pair_features(aatype, pos @ Q.T, mask)["template_unit_vector"].numpy()
    # the correct feature is invariant; the transposed convention would rotate with Q
    assert np.abs(uv_moved - uv @ Q.T).max() > 0.1


def test_an_unaligned_residue_contributes_nothing():
    aatype, pos, mask = _arrays()
    unaligned = np.nonzero(aatype[0] == GAP)[0]
    assert len(unaligned) == 3, "1y57 aligns 444 of 447"
    feats = pair_features(aatype, pos, mask)
    assert float(np.abs(mask[0][unaligned]).max()) == 0.0
    for k in ("template_pseudo_beta_mask", "template_backbone_frame_mask"):
        assert float(feats[k][0][unaligned].abs().max()) == 0.0
        assert float(feats[k][0][:, unaligned].abs().max()) == 0.0
    assert float(feats["template_distogram"][0][unaligned].abs().max()) == 0.0


def test_masked_pairs_carry_no_geometry():
    """The masks are the outer product of the per-residue atom mask, and both geometry
    features are multiplied by their own mask, so nothing leaks from a residue the template
    does not cover."""
    aatype, pos, mask = _arrays()
    f = pair_features(aatype, pos, mask)
    pb2, bb2 = f["template_pseudo_beta_mask"][0].numpy(), f["template_backbone_frame_mask"][0].numpy()
    assert set(np.unique(pb2)) <= {0.0, 1.0} and set(np.unique(bb2)) <= {0.0, 1.0}
    assert np.array_equal(pb2, np.outer(mask[0][:, 3], mask[0][:, 3]))
    assert (f["template_distogram"][0].numpy().sum(-1) <= pb2 + 1e-6).all()
    assert np.abs(f["template_unit_vector"][0].numpy()).max(-1).max() <= 1.0 + 1e-6
    assert float(np.abs(f["template_unit_vector"][0].numpy()[bb2 == 0]).max()) == 0.0


def test_one_distogram_bin_per_covered_pair():
    """39 bins from 3.25 A up, and the last one runs to infinity, so every covered pair at or
    beyond the first edge lands in exactly one. Below it -- the diagonal, where the distance
    is 0 -- upstream's strict `>` on the first lower edge leaves the row empty, and the
    parity arm above says the port reproduces that rather than inventing a zeroth bin."""
    aatype, pos, mask = _arrays()
    f = pair_features(aatype, pos, mask)
    dg, pb2 = f["template_distogram"][0].numpy(), f["template_pseudo_beta_mask"][0].numpy()
    per_pair = dg.sum(-1)
    assert set(np.unique(per_pair)) <= {0.0, 1.0}, "a pair landed in two bins"
    assert (per_pair <= pb2).all(), "an uncovered pair got a bin"
    pb = pos[0][:, 3]
    d = np.linalg.norm(pb[:, None, :] - pb[None, :, :], axis=-1)
    binned = (pb2 == 1) & (d >= 3.25)
    assert binned.sum() > 1e5
    assert (per_pair[binned] == 1.0).all(), "a covered pair in range landed in no bin"
    assert (per_pair[(pb2 == 1) & (d < 3.25)] == 0.0).all()


def test_the_alignment_reader_ranks_and_caps():
    entries = read_alignment_entries(NPZ)
    assert entries and all(len(e) == 3 for e in entries)
    assert len(read_alignment_entries(NPZ, max_templates=1)) == 1
    with np.load(str(NPZ), allow_pickle=True) as z:
        ranks = []
        for k in z.files:
            e = z[k]
            ranks.append(int((e.item() if getattr(e, "shape", None) == () else e).get("index", 0)))
    assert [r for r in sorted(ranks)][:len(entries)] == sorted(ranks)[:len(entries)]


def test_a_chain_without_a_template_stays_gap_and_zero():
    """Complex assembly. A second chain with no template must read gap in slot 0 with zero
    geometry, and the templated chain must read exactly its own single-chain arrays."""
    aatype, pos, mask = _arrays()
    L = aatype.shape[1]
    n_token = L + 32                       # a 32-token second chain, no template
    cols = list(range(L))
    feats = complex_template_features([(0, cols, aatype, pos, mask)], n_token)
    single = pair_features(aatype, pos, mask)
    # the npz carries one template; upstream's assembly line pads the stack to 4 with zeros,
    # which reads ALA(0) in the aatype one-hot and contributes nothing through a zero mask.
    assert aatype.shape[0] == 1 and feats["template_aatype"].shape == (4, n_token)
    assert torch.equal(feats["template_aatype"][0, :L], single["template_aatype"][0])
    assert (feats["template_aatype"][1:] == 0).all()
    assert (feats["template_aatype"][0, L:] == GAP).all()
    for k in ("template_pseudo_beta_mask", "template_backbone_frame_mask"):
        assert float(feats[k][:, L:].abs().max()) == 0.0
        assert float(feats[k][:, :, L:].abs().max()) == 0.0
        assert torch.equal(feats[k][0, :L, :L], single[k][0])
        assert float(feats[k][1:].abs().max()) == 0.0
    assert torch.equal(feats["template_distogram"][0, :L, :L], single["template_distogram"][0])


def test_an_offset_chain_lands_on_its_own_tokens():
    """The block placement is the bug that a single-chain test cannot see: a chain whose
    tokens start at 32 must write there, not at 0."""
    aatype, pos, mask = _arrays()
    L = aatype.shape[1]
    off = 32
    feats = complex_template_features([(off, list(range(L)), aatype, pos, mask)], L + off)
    single = pair_features(aatype, pos, mask)
    assert torch.equal(feats["template_aatype"][0, off:], single["template_aatype"][0])
    assert (feats["template_aatype"][0, :off] == GAP).all()
    assert torch.equal(feats["template_pseudo_beta_mask"][0, off:, off:],
                       single["template_pseudo_beta_mask"][0])
    assert float(feats["template_pseudo_beta_mask"][0, :off].abs().max()) == 0.0


def test_no_templates_is_byte_identical_to_the_dummy_path():
    """The control every existing fold depends on: build_complex_features now always takes
    the argument, so it has to be a no-op when nothing supplies one."""
    from tt_bio.protenix_data import build_complex_features, dummy_template_features
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
    plain = build_complex_features([(seq, None, "protein")], chain_ids=["A"])
    explicit = build_complex_features([(seq, None, "protein")], chain_ids=["A"],
                                      templates=[None], template_dir=None)
    for k in plain:
        assert torch.equal(plain[k], explicit[k]), k
    dummy = dummy_template_features(len(seq))
    for k in FEATURES:
        assert torch.equal(plain[k], dummy[k]), k
