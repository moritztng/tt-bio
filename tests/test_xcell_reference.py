"""The X-Cell host reference's own guard rails.

`tt_bio/xcell_reference.py` is a reconstruction from the preprint's Appendix A, not a port of
running upstream code, so the things worth asserting are the ones a transcription gets silently
wrong: which layers carry cross-attention, whether the tied head reads the raw or the normalised
gene embedding, whether the reveal ladder of the 4-step loop is actually nested, and whether a
missing prior source is ignored rather than attended to as a row of zeros.
"""

import pytest

torch = pytest.importorskip("torch")

from tt_bio.xcell_reference import (  # noqa: E402
    PRIOR_SOURCES, REVEAL_FRACTIONS, XCELL_MINI, XCell, XCellConfig, predict,
)

D = 64


def _cfg(**kw):
    base = dict(d_model=D, n_layers=6, n_heads=4, vocab_size=200, max_genes=32)
    base.update(kw)
    return XCellConfig(**base)


def _batch(cfg, n=3, g=32, seed=0):
    torch.manual_seed(seed)
    return dict(
        values=torch.rand(n, g) * 6,
        tokens=torch.randint(0, cfg.vocab_size, (n, g)),
        pert_mask=torch.zeros(n, g, dtype=torch.long),
        priors={name: torch.randn(n, dim) for name, dim, _ in PRIOR_SOURCES},
        pert_token=torch.randint(0, cfg.vocab_size, (n,)),
        prior_missing=torch.zeros(n, len(PRIOR_SOURCES) + 1, dtype=torch.bool),
    )


# ---------------------------------------------------------------- architecture

def test_cross_attention_placement_matches_both_published_counts():
    """`l % 3 == 2` is the only rule that gives 4 of 12 AND 11 of 34.

    docs/model.md gives X-Cell Mini 4 cross-attention layers at L=12; the preprint's TTA section
    says X-Cell-Ultra bypasses "all 11 cross-attention blocks" at L=34. One rule, two independent
    data points, so a refactor that shifts the stride or the offset fails here.
    """
    assert XCellConfig(n_layers=12).cross_attn_layers == (2, 5, 8, 11)
    assert len(XCellConfig(n_layers=34).cross_attn_layers) == 11


def test_only_cross_layers_have_a_cross_block():
    cfg = _cfg(n_layers=6)
    m = XCell(cfg)
    have = {i for i, b in enumerate(m.blocks) if b.cross is not None}
    assert have == set(cfg.cross_attn_layers) == {2, 5}


def test_mini_defaults_to_the_legacy_scgpt_block():
    """Mini is scGPT-initialized and docs/model.md gives it Post-LN with a 1x ReLU FFN."""
    assert XCELL_MINI.block == "legacy"
    assert XCELL_MINI.d_ff == XCELL_MINI.d_model
    assert XCELL_MINI.global_residual is False   # A23 ablated OFF: MAE 0.178 -> 0.163


def test_modern_block_defaults_to_swiglu_4d():
    assert _cfg(block="modern").d_ff == 4 * D


def test_a_4d_modern_mini_would_blow_the_published_parameter_budget():
    """The budget argument that picks Mini's block, as a test rather than a comment.

    X-Cell Mini is published at 55M. At L=12, d=512 a modern block costs 4d^2 + 3*d*d_ff, so with
    d_ff = 4d the trunk alone overshoots 55M badly; a 1x FFN does not. This is what rules out
    reading A12-A22's block as Mini's.
    """
    def total(**kw):
        return sum(p.numel() for p in XCell(XCellConfig(**kw)).parameters())

    assert total(block="modern") > 75e6           # d_ff = 4d, measured 84.9M
    assert total() < 55e6                         # legacy 1x, measured 43.0M


def test_sequence_length_is_genes_plus_cls():
    cfg = _cfg()
    m = XCell(cfg)
    b = _batch(cfg)
    h = m.embed(b["values"], b["tokens"], b["pert_mask"])
    assert h.shape == (3, 32 + 1, D)


def test_cls_carries_no_value_or_mask_term():
    """A5 prepends a standalone learned vector; only positions 1..G' get value+mask."""
    cfg = _cfg()
    m = XCell(cfg).eval()
    b = _batch(cfg)
    with torch.no_grad():
        h_a = m.embed(b["values"], b["tokens"], b["pert_mask"])
        h_b = m.embed(b["values"] + 3.0, b["tokens"], b["pert_mask"])
    assert torch.equal(h_a[:, 0], h_b[:, 0])              # CLS untouched
    assert not torch.allclose(h_a[:, 1:], h_b[:, 1:])     # genes moved


def test_value_encoder_clips_at_512():
    """A3 clips inside the encoder, so 512 and 5000 must encode identically."""
    cfg = _cfg()
    m = XCell(cfg).eval()
    lo = torch.full((1, 4), 512.0)
    hi = torch.full((1, 4), 5000.0)
    with torch.no_grad():
        assert torch.allclose(m.embed.value(lo), m.embed.value(hi))


def test_perturbation_context_has_six_tokens_in_table_order():
    cfg = _cfg()
    m = XCell(cfg).eval()
    b = _batch(cfg)
    with torch.no_grad():
        c = m.perturbation_context(b["priors"], b["pert_token"])
    assert c.shape == (3, len(PRIOR_SOURCES) + 1, D)
    # The sixth token is the gene embedding (A11), not a projected external source.
    with torch.no_grad():
        gene = m.embed.gene_norm(m.embed.gene(b["pert_token"]))
    assert torch.allclose(c[:, -1], gene)


def test_tied_head_reads_the_raw_not_the_normalised_embedding():
    """A26 uses e_g^raw. Scaling the embedding table must move the output; if the head read the
    LayerNormed embedding it would be nearly invariant to that scaling, which is the silent bug
    this guards.
    """
    cfg = _cfg()
    m = XCell(cfg).eval()
    b = _batch(cfg)
    with torch.no_grad():
        a = m(**b)
        m.embed.gene.weight.mul_(4.0)
        c = m(**b)
    assert (a - c).abs().max() > 1e-3


# ---------------------------------------------------------------- priors / masking

def test_missing_prior_is_masked_not_attended_as_zeros():
    cfg = _cfg()
    m = XCell(cfg).eval()
    b = _batch(cfg)
    miss = b["prior_missing"].clone()
    miss[:, 1] = True                                  # STRING absent
    zeroed = dict(b)
    zeroed["priors"] = dict(b["priors"], string=torch.zeros_like(b["priors"]["string"]))
    with torch.no_grad():
        masked = m(**{**zeroed, "prior_missing": miss})
        attended = m(**zeroed)
    assert torch.isfinite(masked).all()
    # Masking a zero-imputed source is not the same as attending to it.
    assert (masked - attended).abs().max() > 1e-6


def test_all_priors_missing_is_still_finite():
    """Every key padded is the degenerate case that makes an unguarded softmax produce NaN."""
    cfg = _cfg()
    m = XCell(cfg).eval()
    b = _batch(cfg)
    b["prior_missing"] = torch.ones_like(b["prior_missing"])
    with torch.no_grad():
        out = m(**b)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name,dim,_src", PRIOR_SOURCES)
def test_each_prior_projects_from_its_published_width(name, dim, _src):
    cfg = _cfg()
    m = XCell(cfg)
    assert m.priors[name].fc1.in_features == dim
    with torch.no_grad():
        assert m.priors[name](torch.randn(2, dim)).shape == (2, D)


def test_prior_widths_are_the_published_ones():
    assert dict((n, d) for n, d, _ in PRIOR_SOURCES) == {
        "esm": 5120, "string": 512, "genept": 3072, "depmap": 1150, "cp": 259,
    }


# ---------------------------------------------------------------- diffusion loop

def test_reveal_ladder_is_the_published_one():
    assert REVEAL_FRACTIONS == (0.25, 0.5, 0.75, 1.0)


def test_reveal_sets_are_disjoint_and_cover_every_gene():
    """A.4 selects N_t = {r <= a_t} \\ {r <= a_{t-1}}, so the steps partition the genes."""
    torch.manual_seed(3)
    ranks = torch.rand(4, 256)
    prev, seen = 0.0, torch.zeros(4, 256, dtype=torch.bool)
    for a in REVEAL_FRACTIONS:
        newly = (ranks <= a) & (ranks > prev)
        assert not (newly & seen).any(), "reveal sets must be disjoint"
        seen |= newly
        prev = a
    assert seen.all(), "alpha_4 = 1.0 must reveal every gene"


@pytest.mark.parametrize("steps", [1, 2, 3, 4])
def test_predict_runs_every_step_count(steps):
    cfg = _cfg()
    m = XCell(cfg).eval()
    b = _batch(cfg)
    out = predict(m, b["values"], b["tokens"], b["priors"], b["pert_token"],
                  b["prior_missing"], n_steps=steps,
                  generator=torch.Generator().manual_seed(1))
    assert out.shape == b["values"].shape
    assert torch.isfinite(out).all()


def test_predict_is_deterministic_under_a_shared_generator():
    """Shared RNG draws across backends is the parity rule; the loop must honour a seed."""
    cfg = _cfg()
    m = XCell(cfg).eval()
    b = _batch(cfg)
    args = (m, b["values"], b["tokens"], b["priors"], b["pert_token"], b["prior_missing"])
    a = predict(*args, generator=torch.Generator().manual_seed(11))
    c = predict(*args, generator=torch.Generator().manual_seed(11))
    assert torch.equal(a, c)


def test_predict_rejects_an_out_of_range_step_count():
    cfg = _cfg()
    m = XCell(cfg).eval()
    b = _batch(cfg)
    with pytest.raises(ValueError):
        predict(m, b["values"], b["tokens"], b["priors"], b["pert_token"],
                b["prior_missing"], n_steps=5)


# ---------------------------------------------------------------- config guards

def test_config_rejects_indivisible_head_count():
    with pytest.raises(ValueError):
        XCellConfig(d_model=100, n_heads=8)


def test_config_rejects_an_unknown_block():
    with pytest.raises(ValueError):
        XCellConfig(block="postln")


@pytest.mark.parametrize("block", ["legacy", "modern"])
def test_both_blocks_run(block):
    cfg = _cfg(block=block)
    m = XCell(cfg).eval()
    with torch.no_grad():
        out = m(**_batch(cfg))
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()


def test_global_residual_changes_the_output_when_enabled():
    cfg = _cfg(global_residual=True)
    m = XCell(cfg).eval()
    b = _batch(cfg)
    m2 = XCell(_cfg(global_residual=False))
    m2.load_state_dict(m.state_dict())
    m2.eval()
    with torch.no_grad():
        assert (m(**b) - m2(**b)).abs().max() > 1e-4
