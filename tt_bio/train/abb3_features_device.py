"""ABodyBuilder3's input one-hots, built on the card from the indices they expand.

The pair feature is a ``(micro, n_tok, n_tok, 132)`` float tensor -- 138 MB at micro 4 and 256
tokens, 1.107 GB a step at eight micro-batches -- and every value in it is exactly 0.0 or 1.0.
It is a one-hot of a ``(micro, n_tok)`` residue index, which is 8 KB. Building it on the host and
uploading it pays a gigabyte of PCIe a step to carry information that fits in a page; building it
here ships the 8 KB and expands on the card.

**The expansion is integer, so the result is EXACTLY the reference's, not close.** Every input is
a small non-negative integer, exact in fp32, and every operation between them (a difference, a
clamp, an equality) is exact too. ``tests/test_abb3_features_device.py`` asserts
``torch.equal`` against ``abodybuilder3_reference.single_and_pair_features`` on real SAbDab
structures, not a PCC -- a difference here would be wrong rather than imprecise.

The reference is the falsifier and it is never imported for its VALUES on this path, only for its
two constants and by the test. That is deliberate: ``abodybuilder3_reference.py`` is the float64
reference every accuracy number in this campaign is scored against, so it stays byte-identical and
this module gets a check that cannot drift away from it.

Both one-hots are built as a SUM of two equalities rather than a concatenation. The single feature
is a 21-way amino-acid one-hot beside a 2-way chain one-hot and the pair feature is a 3-way
chain-pair one-hot in front of a 129-way relative position, so in both cases the two halves are
disjoint index ranges of one axis: offsetting the second key and adding is the same tensor without
a last-axis concatenation, which is not tile-legal at 21 + 2 or at 3 + 129.
"""

from __future__ import annotations

import torch
import ttnn

from ..abodybuilder3_reference import ABB3Config, REL_POS_DIM

__all__ = ["input_features_device", "pair_features_device", "single_features_device"]

#: Where the chain-pair one-hot ends and the relative-position one-hot begins.
CHAIN_PAIR_DIM = 3
#: Where the amino-acid one-hot ends and the 2-way heavy/light one-hot begins.
AATYPE_DIM = 21

_ARANGE: dict = {}


def _arange(width: int, device):
    """``[0 .. width)`` as a ``(1, 1, 1, width)`` fp32 row, cached per device and width.

    One small constant per bucket, reused by every micro-batch: the caller builds one of these
    per token count and the token axis has few distinct values, so the cache is bounded by the
    number of buckets rather than by the number of steps.
    """
    key = (id(device), int(width))
    t = _ARANGE.get(key)
    if t is None:
        t = ttnn.from_torch(torch.arange(int(width), dtype=torch.float32).reshape(1, 1, 1, width),
                            layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
        _ARANGE[key] = t
    return t


def _up(t: torch.Tensor, shape, device):
    return ttnn.from_torch(t.to(torch.float32).reshape(*shape).contiguous(),
                           layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)


def _two_hot(key_a, key_b, width: int, device):
    """``one_hot(key_a, width) + one_hot(key_b, width)`` for two keys that never collide.

    ``ttnn.eq`` against a broadcast row returns exactly 1.0 or 0.0, so this is one full-size pass
    per hot channel and no intermediate to round. Measured at 10.7 ms for the pair feature against
    17.5 ms for the arithmetic form ``relu(1 - |arange - key|)``, which is also exact but costs
    four full-size ops per one-hot instead of one.
    """
    ar = _arange(width, device)
    return ttnn.add(ttnn.eq(ar, key_a), ttnn.eq(ar, key_b))


def single_features_device(aatype: torch.Tensor, is_heavy: torch.Tensor, *, device,
                           cfg: ABB3Config | None = None):
    """``(micro, n_tok, c_s)`` on the card: a 21-way residue one-hot beside a 2-way chain one-hot."""
    c_s = (cfg or ABB3Config()).c_s
    b, n = int(aatype.shape[0]), int(aatype.shape[1])
    aa = _up(aatype, (b, 1, n, 1), device)
    hv = _up(is_heavy + AATYPE_DIM, (b, 1, n, 1), device)
    return ttnn.reshape(_two_hot(aa, hv, c_s, device), [b, n, c_s])


def pair_features_device(is_heavy: torch.Tensor, residue_index: torch.Tensor, *, device,
                         cfg: ABB3Config | None = None, rel_pos_dim: int = REL_POS_DIM):
    """``(micro, n_tok, n_tok, c_z)`` on the card, from two ``(micro, n_tok)`` index maps.

    The keys are built directly in ``(micro, n_tok, n_tok, 1)`` -- row index on axis 1, column on
    axis 2 -- so no reshape of a full-size tensor happens between the difference and the one-hot.
    """
    c_z = (cfg or ABB3Config()).c_z
    b, n = int(residue_index.shape[0]), int(residue_index.shape[1])
    ri_row, ri_col = _up(residue_index, (b, n, 1, 1), device), _up(residue_index, (b, 1, n, 1), device)
    hv_row, hv_col = _up(is_heavy, (b, n, 1, 1), device), _up(is_heavy, (b, 1, n, 1), device)

    # `rel[i, j] = residue_index[j] - residue_index[i]`, clamped to +-rel_pos_dim, shifted to
    # [0, 2 * rel_pos_dim] and then past the 3 chain-pair channels.
    rel = ttnn.clip(ttnn.subtract(ri_col, ri_row), -float(rel_pos_dim), float(rel_pos_dim))
    k_rel = ttnn.add(rel, float(rel_pos_dim + CHAIN_PAIR_DIM))
    # light/light 1, heavy/heavy 2, mixed 0.
    k_chain = ttnn.add(ttnn.multiply(ttnn.multiply(hv_row, hv_col), 2.0),
                       ttnn.multiply(ttnn.add(hv_row, -1.0), ttnn.add(hv_col, -1.0)))
    return _two_hot(k_rel, k_chain, c_z, device)


def input_features_device(aatype: torch.Tensor, is_heavy: torch.Tensor,
                          residue_index: torch.Tensor, *, device,
                          cfg: ABB3Config | None = None, rel_pos_dim: int = REL_POS_DIM):
    """Both feature maps, the device counterpart of ``single_and_pair_features``."""
    cfg = cfg or ABB3Config()
    return (single_features_device(aatype, is_heavy, device=device, cfg=cfg),
            pair_features_device(is_heavy, residue_index, device=device, cfg=cfg,
                                 rel_pos_dim=rel_pos_dim))
