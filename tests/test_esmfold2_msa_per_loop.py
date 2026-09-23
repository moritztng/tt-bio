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
