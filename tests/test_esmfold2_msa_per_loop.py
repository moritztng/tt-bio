"""ESMFold2's inference MSA, drawn the way esm 3.4.1 draws it (`_msa_per_loop`).

Upstream masks a fraction of the MSA columns once, keeping the query row, and folds each trunk
loop on a fresh `max_depth`-row subsample that always keeps the query. The GPU reference the MGX
campaign scores against was folded that way, so these pin the properties the port relies on.
"""
import pytest

torch = pytest.importorskip("torch")

from tt_bio.esmfold2_runtime import _msa_per_loop  # noqa: E402


def _kw(L=6, M=40, C=4):
    g = torch.Generator().manual_seed(1)
    return {"msa_oh": torch.rand(1, L, M, C, generator=g) + 0.5,
            "has_deletion": torch.arange(M, dtype=torch.float32).expand(1, L, M).clone(),
            "deletion_value": torch.rand(1, L, M, generator=g),
            "msa_attention_mask": torch.ones(1, L, M), "x_inputs": torch.zeros(1, L, 3)}


def test_shallow_msa_is_one_draw_with_the_query_unmasked():
    torch.manual_seed(0)
    loops = _msa_per_loop(_kw(M=40), 3, 1024, 0.5)
    assert len(loops) == 1
    mask = loops[0]["msa_attention_mask"]
    assert bool(mask[:, :, 0].all())
    masked = mask == 0
    assert masked.any()
    # a masked column is masked in every non-query row, and its one-hot is zeroed
    assert bool((masked[:, :, 1:].all(-1) | ~masked[:, :, 1:].any(-1)).all())
    assert float(loops[0]["msa_oh"][masked].abs().sum()) == 0.0


def test_deep_msa_gets_a_fresh_subsample_per_loop_keeping_the_query():
    torch.manual_seed(0)
    loops = _msa_per_loop(_kw(M=40), 3, 8, 0.0)
    assert len(loops) == 3
    rows = [lp["has_deletion"][0, 0].long().tolist() for lp in loops]
    for r in rows:
        assert len(r) == 8 and r[0] == 0 and r == sorted(r) and len(set(r)) == 8
    assert len({tuple(r) for r in rows}) > 1
    for lp in loops:  # every tensor on the row axis took the same rows
        assert lp["msa_oh"].shape[2] == lp["deletion_value"].shape[2] == 8


def test_the_draw_follows_the_seed():
    a = (torch.manual_seed(3), _msa_per_loop(_kw(), 2, 8, 0.1))[1]
    b = (torch.manual_seed(3), _msa_per_loop(_kw(), 2, 8, 0.1))[1]
    for x, y in zip(a, b):
        assert torch.equal(x["msa_attention_mask"], y["msa_attention_mask"])
        assert torch.equal(x["has_deletion"], y["has_deletion"])


def test_no_masking_and_no_subsample_is_the_input_unchanged():
    kw = _kw(M=5)
    (only,) = _msa_per_loop(kw, 3, 1024, 0.0)
    assert all(torch.equal(only[k], kw[k]) for k in kw)


def test_lm_dropout_is_upstreams_training_mode_dropout_and_follows_the_seed():
    from tt_bio.esmfold2_runtime import _lm_dropout

    x = torch.ones(1, 64, 64, 8)
    a = (torch.manual_seed(5), _lm_dropout(x, 0.25))[1]
    b = (torch.manual_seed(5), _lm_dropout(x, 0.25))[1]
    ref = (torch.manual_seed(5), torch.nn.functional.dropout(x, p=0.25, training=True))[1]
    assert torch.equal(a, b)
    assert torch.allclose(a, ref)  # same draw, same 1/(1-p) scale as torch's own dropout
    assert torch.allclose(a.unique(), torch.tensor([0.0, 4.0 / 3.0]))
    assert abs(float((a == 0).float().mean()) - 0.25) < 0.02


def _unpack(w):
    """`_pack_keep`'s inverse on the host, as `_unpack_keep_dev` does it on the device."""
    from tt_bio.esmfold2_runtime import _KEEP_BITS

    assert w.dtype == torch.int32
    return torch.cat([(w >> j) & 1 for j in range(_KEEP_BITS)], dim=1).to(torch.bfloat16)


def test_packed_keep_round_trips_rows_in_order():
    from tt_bio.esmfold2_runtime import _pack_keep

    keep = torch.rand(1, 96, 40, 8) >= 0.25
    w = _pack_keep(keep)
    assert w.shape == (1, 6, 40, 8) and int(w.max()) < 2 ** 16
    assert torch.equal(_unpack(w), keep.to(torch.bfloat16))


def test_prefetched_keep_masks_follow_the_seed_and_keep_one_minus_p():
    from tt_bio.esmfold2_runtime import _KeepDraws

    def draws(seed):
        torch.manual_seed(seed)
        d = _KeepDraws((1, 64, 64, 8), 0.25, 3)
        return [_unpack(d.take()) for _ in range(3)]

    a, b, c = draws(5), draws(5), draws(6)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[0], c[0])
    assert not torch.equal(a[0], a[1])  # a fresh mask every loop
    for m in a:
        assert m.dtype == torch.bfloat16 and set(m.unique().tolist()) == {0.0, 1.0}
        assert abs(float(m.float().mean()) - 0.75) < 0.02


def test_packed_keep_draw_keeps_one_minus_p_per_bit_for_dyadic_p():
    from tt_bio.esmfold2_runtime import _draw_packed_keep, _dyadic

    assert _dyadic(0.25) == (1, 2) and _dyadic(0.375) == (3, 3)
    assert _dyadic(0.1) is None and _dyadic(0.0) is None and _dyadic(1.0) is None
    for p in (0.25, 0.375, 0.5):
        m = _unpack(_draw_packed_keep((1, 256, 64, 32), p, torch.Generator().manual_seed(3)))
        assert m.shape == (1, 256, 64, 32) and set(m.unique().tolist()) == {0.0, 1.0}
        assert abs(float(m.float().mean()) - (1 - p)) < 0.005
        # every bit position of the word carries the same law (no row block biased)
        rows = m.float().mean(dim=(0, 2, 3)).view(16, 16).mean(dim=1)
        assert float((rows - (1 - p)).abs().max()) < 0.01
