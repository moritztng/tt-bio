"""RFD3_DIT_DKNN: the DiT's dense additive mask built on the card (_dit_mask_device) equals the host mask
(_dense_attention_mask of the k-d tree index, uploaded the way LocalTokenTransformer uploads it) bit for bit."""
import numpy as np
import pytest
import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor
from tt_bio.rfd3 import model as M

pytestmark = pytest.mark.device


@pytest.mark.parametrize("batch,sigma", [(1, 0.5), (1, 80.0), (2, 8.0)])
def test_device_mask_equals_host_mask(batch, sigma):
    ensure_p300_mesh_descriptor()
    dev = M.get_device()
    I, K, SEQ = 685, M.RFD3DiffusionModule.DIT_KEYS, M.RFD3DiffusionModule.DIT_SEQ
    rng = np.random.default_rng(11)
    walk = np.cumsum(rng.normal(size=(I, 3)) * 3.8, 0)
    x = torch.from_numpy(np.stack([walk + rng.normal(size=(I, 3)) * sigma for _ in range(batch)]).astype(np.float32))
    f = {"asym_id": torch.zeros(I, dtype=torch.long), "unindexing_pair_mask": torch.zeros(I, I, dtype=torch.bool)}
    if batch > 1:   # a batch keeps the host mask
        assert M._dit_mask_device(f, x, K, SEQ, dev, ttnn.bfloat16) is None
        return
    host = M._tt(M._dense_attention_mask(M._create_attention_indices(f, x, torch.arange(I), K, SEQ)),
                 dev, ttnn.bfloat16)
    got = M._dit_mask_device(f, x, K, SEQ, dev, ttnn.bfloat16)
    assert got is not None
    assert tuple(got.shape) == tuple(host.shape) and got.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(got), ttnn.to_torch(host))
