"""OuterProductMean's whole-token path needs the byte bound its blocked path already has.

The blocked path sizes every row block so the per-block matmul result stays under
`OPM_Z_BUDGET_BYTES`. The whole-token path below it had no bound at all -- it materialises
(tokens, C*D, tokens) and then permutes it, and a ttnn permute is out-of-place, so two of that
tensor are live. Entry to the blocked path was decided purely on token count, so a size just
under that threshold took the unbounded path: 9i3p at 992 tokens with C=D=32 asks for a single
2015363072 B buffer and is refused 704 B per bank short, with 48 % of DRAM free.

The bound may not tighten anything that folds today, so it is anchored on `concat_host_bytes()`
-- one eighth of the part's DRAM, measured on this same target for this same reason -- which
leaves every residue-scale target on the exact whole-token path and leaves Blackhole alone
entirely. Host-only: no device, no network.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

C = D = 32                      # OpenDDE / Protenix OuterProductMean latent widths
WH_DRAM = 12 * 1073741792       # a 12.0 GiB Wormhole Galaxy chip, 12 banks
BH_DRAM = 8 * 4278190080        # a 31.875 GiB p150a, 8 banks


def per_row(tokens, c=C, d=D):
    """One token row of the whole-path product, in bytes: (C*D, tokens) in bf16."""
    return c * d * tokens * 2


def whole(tokens, c=C, d=D):
    return tokens * per_row(tokens, c, d)


@pytest.fixture
def wormhole(monkeypatch):
    """A Wormhole Galaxy after `_apply_grid_thresholds`: 1088 tokens, 1.5 GiB of budget."""
    monkeypatch.setattr(T, "SEQ_LEN_MORE_CHUNKING", 1088)
    monkeypatch.setattr(T, "_CONCAT_HOST_BYTES", T._concat_host_budget(WH_DRAM))
    monkeypatch.delenv("TT_BIO_OPM_WHOLE_PATH_BYTES", raising=False)


def test_the_screen_hook_can_leave_only_the_token_arm(wormhole, monkeypatch):
    """The gate has to be removable without editing it, or it cannot be priced.

    main's `_with_dram_narrowing` recovers this op's refusal reactively and folded 992 on its
    own, so whether gating up front earns its cost is measurable only if a run can turn the
    byte arm off. With it off the token arm alone decides, which is exactly main's behaviour.
    """
    assert T._opm_needs_row_blocks(992, per_row(992))          # byte arm binds by default
    monkeypatch.setenv("TT_BIO_OPM_WHOLE_PATH_BYTES", str(1 << 62))
    assert not T._opm_needs_row_blocks(992, per_row(992))      # only the token arm is left
    assert T._opm_needs_row_blocks(1089, per_row(1089))        # and it still binds above 1088


def test_the_refused_shape_is_the_one_the_token_test_let_through(wormhole):
    """The negative control: without the byte test 992 tokens takes the unbounded path.

    If the token threshold already caught this shape the byte test would be dead weight, and if
    the whole-path product were not the refused number the fix would be aimed at the wrong op.
    """
    assert 992 <= T.SEQ_LEN_MORE_CHUNKING          # the token test alone says whole path
    assert whole(992) == 2015363072                # exactly the buffer the device refused
    assert T._opm_needs_row_blocks(992, per_row(992))


def test_the_budget_is_what_moves_it_and_it_moves_at_887_tokens(wormhole):
    """1.5 GiB of budget puts the boundary at 887 tokens, so name it rather than infer it."""
    assert T.concat_host_bytes() == 1536 * 2 ** 20
    assert not T._opm_needs_row_blocks(886, per_row(886))
    assert T._opm_needs_row_blocks(887, per_row(887))


@pytest.mark.parametrize("tokens", [128, 256, 384, 512, 640, 768, 800, 886])
def test_every_residue_scale_target_keeps_the_exact_whole_path(wormhole, tokens):
    """Nothing that folds today changes path, which is the whole constraint on this bound."""
    assert whole(tokens) <= T.concat_host_bytes()
    assert not T._opm_needs_row_blocks(tokens, per_row(tokens))


@pytest.mark.parametrize("tokens", [887, 960, 992, 1024, 1056, 1088])
def test_the_marginal_band_takes_row_blocks(wormhole, tokens):
    """887 to 1088 is the band the token test used to hand the unbounded path.

    Every one of these asks for more than 1.5 GiB in a single buffer and needs two of them live
    across the permute, on a part with 12.0 GiB total and a trunk that has churned its address
    space.
    """
    assert whole(tokens) > T.concat_host_bytes()
    assert T._opm_needs_row_blocks(tokens, per_row(tokens))


@pytest.mark.parametrize("tokens", [887, 960, 992, 1024, 1056, 1088])
def test_blackhole_is_unchanged_at_every_token_count_wormhole_newly_blocks(monkeypatch, tokens):
    """Neutrality by construction, not by clamp: a p150a's eighth is 3.98 GiB.

    The whole-path product tops out at 2.28 GiB over this band, so no p150a fold changes path,
    and `_apply_grid_thresholds` returns early on a full-size grid so the token test is the
    1536 baseline there.
    """
    monkeypatch.setattr(T, "SEQ_LEN_MORE_CHUNKING", 1536)
    monkeypatch.setattr(T, "_CONCAT_HOST_BYTES", T._concat_host_budget(BH_DRAM))
    assert T.concat_host_bytes() > whole(tokens)
    assert not T._opm_needs_row_blocks(tokens, per_row(tokens))


def test_the_token_test_still_binds_above_it(wormhole):
    """A deep-MSA model with narrow latents can clear the byte test and must still block."""
    narrow = per_row(1120, c=8, d=8)
    assert whole(1120, c=8, d=8) < T.concat_host_bytes()   # the byte test alone would not
    assert T._opm_needs_row_blocks(1120, narrow)
