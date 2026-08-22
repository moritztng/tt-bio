"""The two checkpoint-keyed MSA featurizer fixes upstream shipped with OpenBind (v0.5.0).

B1 `deletion_value` scale: preview2 emits atan(d/3) * 2/acos(0) * 2 = atan(d/3) * 8/pi,
v0.5.0 emits the AF3-spec atan(d/3) * 2/pi. Exactly 4x apart.

B2 MSA `profile` column index: preview2 builds it with np.repeat where the row-major
ravel() above it needs np.tile, so the per-column profile is permuted for any MSA deeper
than one row. `profile` is live -- it is concatenated into s_input (worker.py:
[ai, restype, profile, deletion_mean]).

Both are keyed off the checkpoint rather than corrected outright: preview2 trained 155k
steps on the uncorrected features, so handing it the fixed ones is an input-distribution
shift on a shipped, parity-gated model. That makes "preview2 is bit-identical" as much a
requirement here as "OpenBind matches upstream", and both directions are asserted.

The profile legs compare against captures from the two REAL upstream trees, taken by
scripts/ob0_profile_capture.py run once per tree (they both expose a top-level
`openfold3`, so they cannot share a process):

    PYTHONPATH=<refdeps> OF3_TREE=/tmp/pin_of3 OUT=/tmp/prof_pin.npy <refpy> \\
        scripts/ob0_profile_capture.py
    PYTHONPATH=<refdeps> OF3_TREE=~/ob0_upstream OUT=/tmp/prof_v050.npy <refpy> \\
        scripts/ob0_profile_capture.py

where /tmp/pin_of3 is `git archive 72fc3a95` (tt-bio's vendored pin) from the upstream
clone. Missing captures skip rather than fail.
"""
import math
import os

import numpy as np
import pytest
import torch

_PIN = "/tmp/prof_pin.npy"
_V050 = "/tmp/prof_v050.npy"

# Same construction as scripts/ob0_profile_capture.py. n_rows != n_cols and n_rows > 1:
# np.tile and np.repeat agree at n_rows == 1, which is why a single-sequence fold cannot
# see this fix at all.
_N_ROWS, _N_COLS = 7, 24
_ALPHABET = np.array(list("ACDEFGHIKLMNPQRSTVWY-"), dtype="<U1")


def _msa():
    rng = np.random.default_rng(0)
    return _ALPHABET[rng.integers(0, len(_ALPHABET), size=(_N_ROWS, _N_COLS))]


def _profile(af3_spec):
    from tt_bio._vendor.openfold3.core.data.primitives.sequence.msa import (
        calculate_profile,
    )
    from tt_bio._vendor.openfold3.core.data.resources.residues import MoleculeType

    return np.asarray(calculate_profile(
        msa_array=_msa(), molecule_type=MoleculeType.PROTEIN, chunk_size=1000,
        af3_spec_columns=af3_spec))


# --------------------------------------------------------------------------
# B1: deletion_value scale
# --------------------------------------------------------------------------

def _deletion_value(af3_spec, deletion_matrix):
    """The exact expression each branch of the vendored featurizer evaluates."""
    d = deletion_matrix
    if af3_spec:
        return (torch.atan(d / 3.0) * (2.0 / torch.pi)).to(torch.float32)
    return torch.atan(d / 3.0) * (2.0 / torch.acos(torch.zeros(1)) * 2).to(torch.float32)


def test_deletion_value_scales_are_exactly_4x_apart():
    d = torch.arange(0, 40, dtype=torch.int64).reshape(5, 8)
    prev2 = _deletion_value(False, d)
    ob = _deletion_value(True, d)
    nz = d != 0
    ratio = (prev2[nz] / ob[nz]).double()
    assert torch.allclose(ratio, torch.full_like(ratio, 4.0), rtol=0, atol=1e-6), \
        f"expected exactly 4x, got {ratio.min():.9f}..{ratio.max():.9f}"
    # and the AF3-spec branch really is 2/pi, not merely "4x smaller than what we had"
    expect = (torch.atan(d / 3.0) * (2.0 / math.pi)).to(torch.float32)
    assert torch.equal(ob, expect)


def test_deletion_value_is_zero_for_both_at_depth_one():
    """Why a single-sequence fold cannot see this fix: an all-zero deletion matrix maps
    to zero under either scale, so the smoke folds already run prove nothing about B1."""
    d = torch.zeros(1, 24, dtype=torch.int64)
    assert torch.equal(_deletion_value(False, d), _deletion_value(True, d))


def test_featurizer_config_default_is_preview2():
    """The default must be the preview2 scale, or every existing OF3 fold moves."""
    from tt_bio._vendor.openfold3.core.data.pipelines.featurization.msa import (
        MsaFeaturizerOF3Config,
    )
    cfg = MsaFeaturizerOF3Config(max_rows=16384, max_rows_paired=8191,
                                 subsample_with_bands=False)
    assert cfg.af3_spec_deletion_value is False


# --------------------------------------------------------------------------
# B2: MSA profile column index, against both real upstream trees
# --------------------------------------------------------------------------

def test_profile_flag_actually_changes_the_answer():
    """Guards the gate itself: if the two branches agreed, the tests below would pass on
    an implementation that ignored the flag."""
    a, b = _profile(False), _profile(True)
    assert not np.array_equal(a, b)
    assert np.abs(a - b).max() > 0.1, f"only {np.abs(a - b).max():.4f} apart"


@pytest.mark.skipif(not os.path.exists(_PIN), reason=f"{_PIN} missing")
def test_profile_preview2_matches_the_vendored_pin_bit_exactly():
    """preview2 behaviour is unchanged by the OpenBind work. Bit-exact, not approximate:
    a shipped parity-gated model must not move at all."""
    ref = np.load(_PIN)
    got = _profile(False)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), f"max|diff| {np.abs(got - ref).max()}"


@pytest.mark.skipif(not os.path.exists(_V050), reason=f"{_V050} missing")
def test_profile_openbind_matches_upstream_v050_bit_exactly():
    """OpenBind reproduces upstream v0.5.0 exactly.

    This also settles the pass-1 claim that the v0.5.0 `map_str_array_to_idx_array`
    rewrite (a cached 256-entry LUT over the low byte of each <U1 cell) is behaviourally
    equivalent to the pin's implementation: tt-bio still runs the pin's version, so a
    bit-exact match here can only hold if the rewrite changed nothing.
    """
    ref = np.load(_V050)
    got = _profile(True)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), f"max|diff| {np.abs(got - ref).max()}"


@pytest.mark.skipif(not (os.path.exists(_PIN) and os.path.exists(_V050)),
                    reason="both upstream captures needed")
def test_the_two_upstream_trees_disagree_on_this_input():
    """The premise of the two tests above: they are comparing against different things."""
    assert not np.array_equal(np.load(_PIN), np.load(_V050))


def test_msa_settings_default_is_preview2():
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.dataset_config_components import (  # noqa: E501
        MSASettings,
    )
    assert MSASettings().af3_spec_profile_columns is False


# --------------------------------------------------------------------------
# The flag reaching real features, on a real committed multi-row MSA
# --------------------------------------------------------------------------

_7XI5_SEQ = ("MSSATPDPAEILTARKAVGLSQTAAAALVHSSLRTWQQWEAGDRRMHPGLWELFLLKTQLPSPSS")
_7XI5_MSA = ("docs/implementation-parity-data/ref-fixtures/openfold3/7xi5/"
             "msa-bench-notmpl_200step_5sample_4cycle_fp32cpu/msa_A")


def _features(openbind, msa_dir, msa_settings=None):
    import json
    import tempfile

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (  # noqa: E501
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import build_openfold3_features

    q = {"queries": {"q": {
        "query_name": "q", "use_msas": True, "use_paired_msas": False,
        "use_main_msas": True, "covalent_bonds": None,
        "chains": [{"molecule_type": "PROTEIN", "chain_ids": ["A"],
                    "sequence": _7XI5_SEQ,
                    "main_msa_file_paths": [str(msa_dir)]}]}}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(q, fh)
        path = fh.name
    # The vendored conformer featurizer draws its RDKit ETKDG seed from python's global
    # `random` (conformer.py: "we set a random seed here"), so ref_pos moves between two
    # otherwise identical calls unless it is pinned -- worker.py pins it to the fold seed
    # for exactly this reason. Without this line the comparison below reports ref_pos as
    # having moved, which is a property of this harness and not of the flag.
    import random as _pyrandom

    torch.manual_seed(0)
    np.random.seed(0)
    _pyrandom.seed(0)
    query = next(iter(InferenceQuerySet.from_json(path).queries.values()))
    return build_openfold3_features(query, openbind=openbind,
                                    msa_settings=msa_settings)


def _moved(a, b):
    """Same-shape tensor keys whose contents differ. Shape changes are reported separately:
    the dedup removes MSA rows, so the row-indexed keys cannot be compared elementwise."""
    return sorted(k for k in a
                  if torch.is_tensor(a[k]) and torch.is_tensor(b.get(k))
                  and a[k].shape == b[k].shape and not torch.equal(a[k], b[k]))


def _settings(**over):
    from tt_bio.openfold3_data import inference_msa_settings
    s = inference_msa_settings(openbind=False)
    for k, v in over.items():
        setattr(s, k, v)
    return s


@pytest.mark.skipif(not os.path.isdir(_7XI5_MSA), reason=f"{_7XI5_MSA} missing")
def test_b1_deletion_value_is_exactly_4x_on_a_real_msa():
    """B1 alone, on the committed 7XI5 benchmark MSA (a real multi-row stockholm set, not
    a hand-made fixture). Pinning both calls to preview2 MSA settings isolates the
    deletion scale: it is the one fix that does not live on MSASettings, so it is the only
    thing the `openbind` argument still selects here."""
    prev2 = _features(False, _7XI5_MSA, _settings())
    ob = _features(True, _7XI5_MSA, _settings())

    assert prev2["deletion_value"].shape[0] > 1, "fixture is not multi-row"
    assert _moved(prev2, ob) == ["deletion_value"], _moved(prev2, ob)

    nz = prev2["deletion_value"] != 0
    assert nz.any(), "fixture records no deletions, so B1 is untested by it"
    ratio = (prev2["deletion_value"][nz] / ob["deletion_value"][nz]).double()
    assert torch.allclose(ratio, torch.full_like(ratio, 4.0), rtol=0, atol=1e-5), \
        f"deletion_value ratio {ratio.min():.7f}..{ratio.max():.7f}, expected 4"


@pytest.mark.skipif(not os.path.isdir(_7XI5_MSA), reason=f"{_7XI5_MSA} missing")
def test_b2_profile_is_a_within_column_permutation_on_a_real_msa():
    """B2 alone: flip only the profile column index, so `profile` is the single mover and
    its per-column sums survive. Checking the sums pins this as the column scramble rather
    than unrelated numeric drift."""
    prev2 = _features(False, _7XI5_MSA, _settings())
    ob = _features(False, _7XI5_MSA, _settings(af3_spec_profile_columns=True))

    assert _moved(prev2, ob) == ["profile"], _moved(prev2, ob)
    assert torch.allclose(prev2["profile"].sum(-1), ob["profile"].sum(-1), atol=1e-5)


@pytest.mark.skipif(not os.path.isdir(_7XI5_MSA), reason=f"{_7XI5_MSA} missing")
def test_b5_dedup_shrinks_the_main_msa_and_moves_the_averages():
    """B5 alone: the dedup drops duplicate rows of the concatenated main MSA, so every
    row-indexed feature gets shorter, and `profile` / `deletion_mean` move because they
    average over that array. Nothing else may move."""
    prev2 = _features(False, _7XI5_MSA, _settings())
    ob = _features(False, _7XI5_MSA, _settings(af3_spec_main_msa_dedup=True))

    row_keyed = ["msa", "msa_mask", "deletion_value", "has_deletion"]
    for k in row_keyed:
        assert ob[k].shape[0] < prev2[k].shape[0], f"{k} did not shrink"
    assert len({ob[k].shape[0] for k in row_keyed}) == 1, "row-indexed keys disagree on depth"
    assert _moved(prev2, ob) == ["deletion_mean", "profile"], _moved(prev2, ob)


@pytest.mark.skipif(not os.path.isdir(_7XI5_MSA), reason=f"{_7XI5_MSA} missing")
def test_b3_uppercase_is_a_noop_for_these_alignments():
    """B3 is a no-op unless the parsed array actually holds lowercase. The a3m parser
    deletes lowercase into the deletion matrix before building the array, so only .sto and
    pre-parsed .npz can carry any. Recorded as a measurement, not an assumption: if a
    future fixture does carry lowercase this test is the thing that notices."""
    base = _features(False, _7XI5_MSA, _settings())
    upper = _features(False, _7XI5_MSA, _settings(af3_spec_uppercase_msa=True))
    assert _moved(base, upper) == [], _moved(base, upper)


@pytest.mark.skipif(not os.path.isdir(_7XI5_MSA), reason=f"{_7XI5_MSA} missing")
def test_the_whole_flag_moves_every_fix_it_owns_and_nothing_else():
    """End of the plumbing: the leg that would catch the flag being accepted and then
    dropped somewhere between build_openfold3_features and the four vendored call sites.
    The single-sequence folds cannot catch it -- every fix is an identity at MSA depth 1."""
    prev2 = _features(False, _7XI5_MSA)
    ob = _features(True, _7XI5_MSA)

    assert ob["msa"].shape[0] < prev2["msa"].shape[0], "dedup did not reach the features"
    assert _moved(prev2, ob) == ["deletion_mean", "profile"], _moved(prev2, ob)
