"""ESMC and SaProt never hand ttnn's SDPA a None mask.

The unmasked op returns the wrong attention at some chunk configs on an 8x9 grid, so an aligned
token axis (nothing to pad) must still carry an all-zero mask. See esmc.bucket_token_axis.
"""
import pytest

torch = pytest.importorskip("torch")
esmc = pytest.importorskip("tt_bio.esmc")


@pytest.mark.parametrize("L", [32, 128, 256])
def test_an_aligned_length_still_gets_a_zero_mask(L):
    ids = torch.zeros(2, L, dtype=torch.long)
    tokens, mask, key_valid, _, n = esmc.bucket_token_axis(ids, bucket=32)
    assert tokens is ids and n == L and key_valid is None
    assert mask.shape == (2, L, L) and torch.count_nonzero(mask) == 0


def test_a_caller_mask_at_an_aligned_length_is_passed_through():
    ids = torch.zeros(1, 64, dtype=torch.long)
    own = torch.randn(1, 64, 64)
    assert esmc.bucket_token_axis(ids, own, bucket=32)[1] is own


def test_a_ragged_length_masks_only_the_padded_keys():
    ids = torch.zeros(1, 126, dtype=torch.long)
    _, mask, key_valid, _, n = esmc.bucket_token_axis(ids, bucket=32)
    assert n == 126 and mask.shape == (1, 128, 128)
    assert torch.all(mask[..., :126] == 0) and torch.all(torch.isinf(mask[..., 126:]))
    assert torch.all(key_valid[:, :, 126:] == 0)
