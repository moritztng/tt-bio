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
    PRIOR_SOURCES, REVEAL_FRACTIONS, XCELL_MINI, XCELL_ULTRA, XCell, XCellConfig, predict,
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

def test_cross_attention_indices_are_the_published_ones():
    """Table A2 publishes Mini's indices literally as 2, 5, 8, 11, and Ultra's count as 11.

    `l % 3 == 2` reproduces both, so a refactor that shifts the stride or the offset fails here.
    """
    assert XCellConfig(n_layers=12).cross_attn_layers == (2, 5, 8, 11)
    assert len(XCellConfig(n_layers=34).cross_attn_layers) == 11


def test_cross_attention_indices_can_be_pinned_explicitly():
    cfg = XCellConfig(n_layers=8, cross_attn_indices=(1, 7))
    assert cfg.cross_attn_layers == (1, 7)
    m = XCell(cfg)
    assert {i for i, b in enumerate(m.blocks) if b.cross is not None} == {1, 7}


def test_only_cross_layers_have_a_cross_block():
    cfg = _cfg(n_layers=6)
    m = XCell(cfg)
    have = {i for i, b in enumerate(m.blocks) if b.cross is not None}
    assert have == set(cfg.cross_attn_layers) == {2, 5}


def test_mini_matches_its_published_configuration():
    """Every X-Cell (55M) column of Table A2 that this module models."""
    c = XCELL_MINI
    assert (c.n_layers, c.d_model, c.n_heads, c.d_head) == (12, 512, 8, 64)
    assert c.d_ff == 512 == c.d_model          # "512 (1 x d)"
    assert c.block == "legacy"                 # Post-LN (LayerNorm), ReLU FFN
    assert c.output_head == "mlp"              # NOT tied embeddings; tying is Ultra's
    assert c.cross_attn_layers == (2, 5, 8, 11)
    assert c.global_residual is False          # "Hidden residual: No"


def test_ultra_matches_its_published_configuration():
    c = XCELL_ULTRA
    assert (c.n_layers, c.d_model, c.n_heads, c.d_head) == (34, 2560, 40, 64)
    assert c.d_ff == 10240 == 4 * c.d_model    # "10240 (4 x d)"
    assert c.block == "modern"                 # Pre-LN (RMSNorm), SwiGLU
    assert c.output_head == "tied"
    assert len(c.cross_attn_layers) == 11
    assert c.global_residual is False


def test_mini_has_attention_bias_and_no_qk_norm():
    """Table A2: Mini has attention bias Yes, QK-Norm No; Ultra is the reverse."""
    mini = XCell(_cfg(block="legacy")).blocks[0]
    assert mini.attn.q.bias is not None
    assert mini.attn.q_norm is None
    modern = XCell(_cfg(block="modern")).blocks[0]
    assert modern.attn.q.bias is None
    assert modern.attn.q_norm is not None


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


    assert total(block="modern", d_model=512, n_layers=12, n_heads=8,
                 vocab_size=19_400) > 75e6        # d_ff = 4d, measured 84.9M
    assert total(d_model=512, n_layers=12, n_heads=8,
                 vocab_size=19_400) < 55e6        # legacy 1x, measured 43.0M


def test_ultra_reconstructs_its_published_parameter_count():
    """The strongest single check on the whole reconstruction.

    X-Cell-Ultra is published at 4.87B. This module's Ultra config measures 4.860B, 0.2% off,
    which it can only do if the block shape, d_ff = 4d, L = 34, the 11 cross-attention blocks and
    the tied head are all right together.
    """
    n = sum(p.numel() for p in XCell(XCELL_ULTRA).parameters())
    assert 4.80e9 < n < 4.95e9, f"{n/1e9:.3f}B is not the published 4.87B"


def test_cross_blocks_carry_their_own_ffn():
    """A22 is a SECOND SwiGLU, distinct from A19's, and Ultra's parameter count proves it.

    Dropping the cross-block FFN would cost ~0.86B of Ultra's 4.87B, i.e. land ~18% low. So this
    is not a stylistic reading of the appendix; the published count discriminates it.
    """
    block = XCell(XCellConfig(n_layers=3, d_model=64, n_heads=4, vocab_size=50,
                              block="modern")).blocks[2]
    assert block.cross is not None
    assert hasattr(block, "cross_ffn"), "A22 FFN missing from the cross-attention block"
    assert block.cross_ffn is not block.ffn


def test_mlp_head_has_no_per_gene_bias_table_and_tied_head_does():
    assert not hasattr(XCell(_cfg(output_head="mlp")).decoder, "gene_bias")
    assert hasattr(XCell(_cfg(output_head="tied")).decoder, "gene_bias")


@pytest.mark.parametrize("head", ["mlp", "tied"])
def test_both_output_heads_run(head):
    cfg = _cfg(output_head=head)
    m = XCell(cfg).eval()
    with torch.no_grad():
        out = m(**_batch(cfg))
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()


def test_config_rejects_an_unknown_output_head():
    with pytest.raises(ValueError):
        XCellConfig(output_head="linear")


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
    this guards. Only the tied head does this, so the test pins output_head="tied".
    """
    cfg = _cfg(output_head="tied")
    m = XCell(cfg).eval()
    b = _batch(cfg)
    with torch.no_grad():
        a = m(**b)
        m.embed.gene.weight.mul_(4.0)
        c = m(**b)
    assert (a - c).abs().max() > 1e-3


# ------------------------------------------------- cross-validation against a real released impl

def test_legacy_block_matches_torchs_own_postln_encoder_layer():
    """The one part of this reconstruction with an EXTERNAL reference, so it gets a test.

    Everything else here is checked against itself or against a published parameter count. But
    X-Cell Mini's block is scGPT's, and scGPT's encoder layer IS
    `torch.nn.TransformerEncoderLayer(norm_first=False, activation="relu")`. So the Post-LN claim
    can be checked against PyTorch's own implementation rather than against our reading of it.
    Measured: 4.8e-07, fp32 noise.
    """
    torch.manual_seed(0)
    from tt_bio.xcell_reference import LegacyBlock

    d, h, s_len, n = 64, 4, 24, 3
    cfg = _cfg(d_model=d, n_heads=h, block="legacy")
    ours = LegacyBlock(cfg, cross=False).eval()
    theirs = torch.nn.TransformerEncoderLayer(
        d_model=d, nhead=h, dim_feedforward=cfg.d_ff, dropout=0.0,
        activation="relu", batch_first=True, norm_first=False).eval()
    with torch.no_grad():
        theirs.self_attn.in_proj_weight.copy_(
            torch.cat([ours.attn.q.weight, ours.attn.k.weight, ours.attn.v.weight], 0))
        theirs.self_attn.in_proj_bias.copy_(
            torch.cat([ours.attn.q.bias, ours.attn.k.bias, ours.attn.v.bias], 0))
        theirs.self_attn.out_proj.weight.copy_(ours.attn.o.weight)
        theirs.self_attn.out_proj.bias.copy_(ours.attn.o.bias)
        theirs.linear1.weight.copy_(ours.fc1.weight); theirs.linear1.bias.copy_(ours.fc1.bias)
        theirs.linear2.weight.copy_(ours.fc2.weight); theirs.linear2.bias.copy_(ours.fc2.bias)
        theirs.norm1.weight.copy_(ours.attn_norm.weight)
        theirs.norm1.bias.copy_(ours.attn_norm.bias)
        theirs.norm2.weight.copy_(ours.ffn_norm.weight)
        theirs.norm2.bias.copy_(ours.ffn_norm.bias)
        x = torch.randn(n, s_len, d)
        assert (ours(x, None, None) - theirs(x)).abs().max() < 1e-5


def test_modern_block_does_not_match_a_postln_layer():
    """The negative control for the test above: if both matched, the two blocks would not
    actually differ and `XCellConfig.block` would be decorative."""
    torch.manual_seed(0)
    d, h, s_len, n = 64, 4, 24, 3
    modern = XCell(_cfg(d_model=d, n_heads=h, block="modern")).blocks[0].eval()
    theirs = torch.nn.TransformerEncoderLayer(
        d_model=d, nhead=h, dim_feedforward=4 * d, dropout=0.0,
        activation="relu", batch_first=True, norm_first=False).eval()
    x = torch.randn(n, s_len, d)
    with torch.no_grad():
        assert (modern(x, None, None) - theirs(x)).abs().max() > 0.1


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
