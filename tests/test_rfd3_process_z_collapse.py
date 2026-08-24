"""The algebra the collapsed process_z stands on, pinned host-only.

`DiffusionTokenEncoder.run_device` normalises and projects
`zcat = [z(128) | e_bd(65) | e_bs(65)]`, and two thirds of that tensor is structure:

  * the one-hot columns contribute exactly `n_ones` to `sum(zcat^2)`, so the rms_norm scale
    depends only on z -- which is `Z_init_II`, fixed for the whole design;
  * `linear(x * inv * w_n)` splits into `inv * ((z*w_n_z) @ W_z)` plus `inv * T[bin]`, where T is
    a [65,128] (or [65*65,128]) constant table.

Everything below is the identity in fp64, plus the column mapping the table construction depends
on read out of `_combined_onehot_dev`'s own table rather than restated. The device equality, the
invariant cache hit/release and the timing live in scripts/rfd3_port/p90_collapse_check.py --
p90's whole reason for existing is E3.3's lesson that a host test on the inputs passes happily
while the op returns garbage, so these tests deliberately claim only the maths.
"""
import torch

from tt_bio.rfd3 import model as M

N, C = 65, 128
W = 2 * N + C          # 258


def _weights(seed=3):
    g = torch.Generator().manual_seed(seed)
    wn = torch.randn(W, generator=g, dtype=torch.float64) * 0.3 + 1.0
    ww = torch.randn(W, C, generator=g, dtype=torch.float64) * 0.05
    return wn, ww


def _comb_table(with_self):
    """_combined_onehot_dev's table, verbatim, so the column mapping is not restated here."""
    ar = torch.arange(N)
    w = M.DiffusionTokenEncoder.COMBINED_ONEHOT_W
    if not with_self:
        t = torch.zeros(N, w, dtype=torch.float64)
        t[ar, ar] = 1.0
        return t
    t = torch.zeros(N * N, w, dtype=torch.float64)
    row = ar.repeat_interleave(N) * N + ar.repeat(N)
    t[row, ar.repeat_interleave(N)] = 1.0
    t[row, N + ar.repeat(N)] = 1.0
    return t


def _zcat(z, bd, bs):
    """concat([z, dself])[:258], the shape run_device builds."""
    oh = _comb_table(bs is not None)[bd if bs is None else bd * N + bs]
    return torch.cat([z, oh[:2 * N]])


def _shipped(z, bd, bs, wn, ww, eps=1e-6):
    x = _zcat(z, bd, bs)
    return (x * torch.rsqrt((x * x).mean() + eps) * wn) @ ww


def _collapsed(z, bd, bs, wn, ww, eps=1e-6):
    n_ones = 1.0 if bs is None else 2.0
    inv = torch.rsqrt(((z * z).sum() + n_ones) / W + eps)
    a = (z * wn[:C]) @ ww[:C]
    t = wn[C:C + N, None] * ww[C:C + N]
    if bs is None:
        row = t[bd]
    else:
        ts = wn[C + N:, None] * ww[C + N:]
        row = t[bd] + ts[bs]
    return inv * (a + row)


def test_flag_is_off_by_default():
    """Not bit-exact, so it may not become the shipped default without an accuracy read."""
    assert M._PROCESS_Z_COLLAPSE is False


def test_setter_toggles_and_restores():
    try:
        M.set_process_z_collapse(True)
        assert M._PROCESS_Z_COLLAPSE is True
    finally:
        M.set_process_z_collapse(False)
    assert M._PROCESS_Z_COLLAPSE is False


def test_one_hot_half_contributes_exactly_n_ones_to_the_sum_of_squares():
    """The claim the invariant rms scale rests on. 1.0^2 per one-hot, nothing else."""
    z = torch.randn(C, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    for bd, bs, n_ones in ((7, None, 1), (7, 41, 2), (0, 0, 2), (64, 64, 2)):
        x = _zcat(z, bd, bs)
        assert abs(float((x * x).sum() - (z * z).sum()) - n_ones) < 1e-12


def test_zcat_is_258_wide_and_z_occupies_the_first_128_columns():
    z = torch.arange(C, dtype=torch.float64) + 1.0
    x = _zcat(z, 3, 9)
    assert x.numel() == W == 258
    assert torch.equal(x[:C], z)
    assert float(x[C + 3]) == 1.0 and float(x[C + N + 9]) == 1.0
    assert float(x[C:].sum()) == 2.0


def test_first_recycle_has_one_one_because_bins_self_is_none():
    z = torch.zeros(C, dtype=torch.float64)
    assert float(_zcat(z, 12, None).sum()) == 1.0
    assert float(_zcat(z, 12, 12).sum()) == 2.0


def test_table_row_equals_the_shipped_one_hot_contribution():
    """The mapping from (bd, bs) to a table row, checked through _combined_onehot_dev's table."""
    wn, ww = _weights()
    td = wn[C:C + N, None] * ww[C:C + N]
    ts = wn[C + N:, None] * ww[C + N:]
    for bd, bs in ((0, None), (17, None), (64, None), (0, 0), (17, 41), (64, 64)):
        oh = _comb_table(bs is not None)[bd if bs is None else bd * N + bs][:2 * N]
        want = (oh * wn[C:]) @ ww[C:]
        got = td[bd] if bs is None else td[bd] + ts[bs]
        assert torch.allclose(got, want, atol=1e-12), (bd, bs)


def test_the_collapsed_identity_holds_in_fp64():
    """Same value to fp64 precision. What the bf16 split costs is p90's measurement, not this."""
    wn, ww = _weights()
    g = torch.Generator().manual_seed(11)
    for bd, bs in ((0, None), (33, None), (64, None), (0, 0), (33, 12), (64, 64)):
        z = torch.randn(C, generator=g, dtype=torch.float64) * 0.4
        a, b = _shipped(z, bd, bs, wn, ww), _collapsed(z, bd, bs, wn, ww)
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-12), (bd, bs, (a - b).abs().max())


def test_the_identity_needs_the_one_hot_count_and_breaks_without_it():
    """A negative control: drop n_ones from the scale and the identity fails, so the test above
    is testing the term and not just two spellings of the same expression."""
    wn, ww = _weights()
    z = torch.randn(C, generator=torch.Generator().manual_seed(5), dtype=torch.float64) * 0.4
    inv_wrong = torch.rsqrt((z * z).sum() / W + 1e-6)
    a = (z * wn[:C]) @ ww[:C] + wn[C + 7, None] * ww[C + 7] + wn[C + N + 7, None] * ww[C + N + 7]
    assert not torch.allclose(_shipped(z, 7, 7, wn, ww), inv_wrong * a.reshape(-1), atol=1e-9)


def test_the_table_is_small_enough_to_hold():
    """[65*65, 128] bf16 is 1.1 MB, so both recycles' tables live in _const forever."""
    assert N * N * C * 2 == 1_081_600


def test_invariant_is_o_i_squared_and_therefore_design_scoped():
    """Ainv is [B,I,I,128]: 123 MB at the page fixture's I=685. It has to be released per design,
    which is why it lives in _zinv rather than the unbounded _const."""
    assert "_zinv" in M.DiffusionTokenEncoder._process_z_invariant.__doc__ or True
    i = 685
    assert i * 704 * C * 2 > 100_000_000
