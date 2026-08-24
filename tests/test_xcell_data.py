"""X-Cell's input pipeline against the preprint's appendix A.10. No device, no weights."""
import numpy as np
import pytest

from tt_bio import xcell_data as X


def test_cp10k_log1p_matches_the_published_formula():
    """A.10.2: x <- 1e4 * x / sum(x), then log(1+x)."""
    raw = np.array([[1.0, 3.0, 6.0], [10.0, 10.0, 30.0]])
    got = X.normalize(raw)
    want = np.log1p(1e4 * raw / raw.sum(axis=1, keepdims=True))
    assert np.allclose(got, want, atol=1e-5)


def test_prenormalized_input_passes_through_untouched():
    """Replogle-Nadig ships CP10k+log1p already; normalising twice is silent and wrong."""
    x = np.array([[0.5, 1.5], [2.0, 0.0]], dtype=np.float32)
    assert np.array_equal(X.normalize(x, pre_normalized=True), x)
    assert not np.allclose(X.normalize(x), x)


def test_all_zero_cell_does_not_divide_by_zero():
    got = X.normalize(np.zeros((2, 4)))
    assert np.isfinite(got).all() and (got == 0).all()


def test_perturbed_gene_is_forced_to_position_zero():
    """A.10.2 forces the knockdown target in at position 1 of the sequence (0 is CLS).

    A subsample that drops the target asks the model to predict a perturbation it cannot see, so
    this holds over many draws rather than one lucky one.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        picked = X.subsample_genes(500, pert_index=137, n_genes=16, rng=rng)
        assert picked[0] == 137
        assert len(set(picked.tolist())) == 16, "subsample must not repeat genes"


def test_subsample_without_a_perturbed_gene_is_still_unique():
    picked = X.subsample_genes(100, None, 40, np.random.default_rng(1))
    assert len(set(picked.tolist())) == 40


def test_subsample_caps_at_the_available_gene_count():
    picked = X.subsample_genes(10, 3, 4000, np.random.default_rng(0))
    assert len(picked) == 10 and picked[0] == 3


@pytest.mark.parametrize("n_cells,set_size", [(64, 64), (65, 64), (10, 64), (200, 64), (7, 3)])
def test_every_set_is_exactly_set_size(n_cells, set_size):
    """A.10.1: incomplete sets are padded by sampling WITH replacement, so none is short."""
    sets = X.build_sets(n_cells, set_size, np.random.default_rng(0))
    assert sets and all(len(s) == set_size for s in sets)
    assert len(sets) == -(-n_cells // set_size)


def test_full_sets_do_not_repeat_cells():
    sets = X.build_sets(128, 64, np.random.default_rng(0))
    for s in sets:
        assert len(set(s.tolist())) == 64


def test_every_cell_appears_at_least_once():
    sets = X.build_sets(100, 64, np.random.default_rng(3))
    seen = set(int(i) for s in sets for i in s)
    assert seen == set(range(100))


def test_prepare_end_to_end_shapes_and_gene_order():
    names = [f"G{i}" for i in range(300)]
    expr = np.random.default_rng(0).random((70, 300)).astype(np.float32) * 20
    sets = X.prepare(expr, names, "G42", n_genes=32, set_size=64,
                     rng=np.random.default_rng(0))
    assert len(sets) == 2                       # 70 cells -> two sets of 64
    for s in sets:
        assert s.values.shape == (64, 32)
        assert s.gene_names[0] == "G42"         # perturbed gene first
        assert s.pert_gene == "G42"
        assert np.isfinite(s.values).all()


def test_prepare_rejects_a_perturbed_gene_the_matrix_does_not_carry():
    names = [f"G{i}" for i in range(10)]
    expr = np.ones((5, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="not in this dataset"):
        X.prepare(expr, names, "NOTAGENE")


def test_prepare_rejects_a_gene_name_length_mismatch():
    with pytest.raises(ValueError, match="gene names"):
        X.prepare(np.ones((4, 6), dtype=np.float32), ["a", "b"], "a")


def test_prepare_rejects_a_non_matrix():
    with pytest.raises(ValueError, match="2-D"):
        X.prepare(np.ones(6, dtype=np.float32), ["a"] * 6, "a")


def test_published_defaults_are_the_published_ones():
    assert X.MAX_GENES == 4000 and X.SET_SIZE == 64 and X.CP10K_TARGET == 1e4


def test_h5ad_without_anndata_says_what_to_install():
    """anndata is optional on purpose; the error has to name the fix rather than trace."""
    try:
        import anndata  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install anndata"):
            X.read_h5ad("nope.h5ad")
    else:
        pytest.skip("anndata is installed in this env")
