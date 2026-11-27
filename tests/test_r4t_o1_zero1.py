"""Host-only unit tests for r4t-perf O1 (ZeRO-1 sharding partition & load balance)."""
from __future__ import annotations

import math

from tt_bio.train.device_optim import _assign_shards
from tt_bio.train.optim import Muon
from tt_bio.train.rfd4_objective import MUON_EXCLUDE


def _rfd4_shapes() -> dict[str, tuple[int, ...]]:
    """Representative RFD4 parameter shape distribution (558 params, 178.4M elems)."""
    spec = [
        ("nn.tr.%d.swish.weight", (6144, 768), 14),
        ("nn.tr.%d.out.weight", (768, 3072), 14),
        ("nn.tr.%d.qkv.weight", (2304, 768), 14),
        ("nn.pu.%d.linx.weight", (768, 768), 36),
        ("nn.tr.%d.adaln.weight", (768, 256), 84),
        ("nn.pu.%d.tri.weight", (256, 384), 56),
        ("nn.pu.%d.linz.weight", (384, 256), 16),
        ("nn.pu.%d.ling.weight", (384, 384), 15),
        ("nn.tr.%d.to_bias.weight", (12, 384), 14),
        ("nn.pu.%d.out1.weight", (768, 384), 7),
        ("nn.pu.%d.out2.weight", (384, 768), 7),
        ("glue.tr_c.%d.sw.weight", (1024, 256), 2),
        ("glue.tr_c.%d.lo.weight", (256, 512), 2),
        ("glue.sequence_readout.1.weight", (147, 768), 1),
        ("nn.ln768.%d.bias", (768,), 151),
        ("nn.ln384.%d.bias", (384,), 52),
        ("nn.ln256.%d.weight", (256,), 44),
        ("nn.qkv.%d.bias", (2304,), 14),
        ("glue.insertion_q_proj.weight", (768, 768), 1),
        ("glue.insertion_k_proj.weight", (768, 768), 1),
        ("glue.init_repr_factory.linear_out.weight", (768, 897), 1),
        ("glue.pair_repr_builder.init_repr_factory.linear_out.weight", (384, 107), 1),
        ("glue.cond_factory.linear_out.weight", (256, 609), 1),
        ("glue.pair_repr_builder.cond_factory.linear_out.weight", (256, 512), 1),
    ]
    out = {}
    for pat, shp, count in spec:
        if "%d" in pat:
            for i in range(count):
                out[pat % i] = shp
        else:
            out[pat] = shp
    return out


def test_zero1_partition_disjoint_complete_and_deterministic():
    shapes = _rfd4_shapes()
    mu = Muon(exclude=MUON_EXCLUDE, adjust_lr_fn="match_rms_adamw", ns_steps=5)
    muon_names = frozenset(n for n, s in shapes.items() if mu.takes(n, s))

    for world in (1, 2, 4, 8, 32):
        owner = _assign_shards(shapes, muon_names, world)
        assert set(owner) == set(shapes)
        assert all(0 <= r < world for r in owner.values())
        # Deterministic regardless of input dict insertion order
        rev = dict(reversed(list(shapes.items())))
        assert _assign_shards(rev, muon_names, world) == owner


def test_zero1_32_muon_flops_well_balanced():
    shapes = _rfd4_shapes()
    mu = Muon(exclude=MUON_EXCLUDE, adjust_lr_fn="match_rms_adamw", ns_steps=5)
    muon_names = frozenset(n for n, s in shapes.items() if mu.takes(n, s))
    owner = _assign_shards(shapes, muon_names, 32)

    flops = [0.0] * 32
    for n in muon_names:
        m, k = min(shapes[n]), max(shapes[n])
        flops[owner[n]] += float(10 * m * m * k + 4 * m * m * m)
    mean_f = sum(flops) / 32.0
    max_ratio = max(flops) / mean_f
    assert max_ratio < 1.15, f"unbalanced 32-way Muon NS FLOPs: max/mean={max_ratio:.3f}"


def test_zero1_broken_unsharded_detects_missing_rank():
    shapes = _rfd4_shapes()
    mu = Muon(exclude=MUON_EXCLUDE, adjust_lr_fn="match_rms_adamw", ns_steps=5)
    muon_names = frozenset(n for n, s in shapes.items() if mu.takes(n, s))
    owner = _assign_shards(shapes, muon_names, 4)
    # Every rank must own at least one Muon and one Adam parameter at world=4
    for r in range(4):
        assert any(owner[n] == r for n in muon_names)
        assert any(owner[n] == r for n in shapes if n not in muon_names)
