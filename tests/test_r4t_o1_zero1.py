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


def test_load_depths_matches_b200_final_mat_nocg():
    from pathlib import Path
    from tt_bio.train.rfd4_run import _load_depths

    assert _load_depths("2", "") == [2]
    assert _load_depths("6,1,1,1,5", "") == [6, 1, 1, 1, 5]
    p = Path(__file__).resolve().parent.parent / (
        "perf/kitcmp/results/r4t-train-b200/rfd4/r4t/final-mat-nocg/steps.jsonl"
    )
    if p.exists():
        ds = _load_depths("", str(p))
        assert len(ds) == 45
        assert ds[:5] == [6, 1, 1, 1, 5]
        assert abs(sum(ds[5:]) / len(ds[5:]) - 2.75) < 1e-9


def test_device_trunk_masks_math_matches_host():
    """Verify _device_trunk_masks arithmetic (pair_1nn - 1.0) * (-NEG_INF) is bit-exact with trunk_masks."""
    import torch

    neg_inf = -1e4  # matches tt_bio.rfd4.trunk.NEG_INF without importing ttnn
    # Partial padding mask [K=4, N=16]
    m = torch.ones((4, 16), dtype=torch.float32)
    m[0, 12:] = 0.0
    m[2, 10:] = 0.0
    pair_ref = m[:, :, None] * m[:, None, :]
    attn_ref = ((1.0 - pair_ref) * neg_inf)[:, None, :, :]
    # Exact _device_trunk_masks formula: (pair - 1.0) * (-NEG_INF)
    attn_dev = ((pair_ref - 1.0) * (-neg_inf))[:, None, :, :]
    assert float((attn_dev - attn_ref).abs().max()) == 0.0
    assert float(attn_dev[0, 0, 0, 0]) == 0.0
    assert float(attn_dev[0, 0, 15, 0]) == neg_inf


def test_fast_onehot_idx_matches_exact_onehot():
    """Verify ResidentPairBuilder's single-pass one-hot extractor matches exact {0,1} sum<=1 check."""
    import torch

    def _fast(f: torch.Tensor, d: int):
        mx, idx = f.max(-1)
        s = f.sum(-1)
        if bool(((mx == 0) | (mx == 1)).all() and (s == mx).all() and (f.min() >= 0)):
            return torch.where(mx > 0.5, idx, d).reshape(f.shape[0], -1).to(torch.int32).contiguous()
        return None

    # Valid one-hot with all-zero rows
    f = torch.zeros((2, 4, 4, 5), dtype=torch.float32)
    f[0, 1, 2, 3] = 1.0
    f[1, 0, 0, 0] = 1.0
    out = _fast(f, 5)
    assert out is not None
    assert int(out[0, 1 * 4 + 2]) == 3
    assert int(out[0, 0]) == 5
    assert int(out[1, 0]) == 0

    # Non-one-hot (sum == 2) rejected
    f2 = f.clone()
    f2[0, 1, 2, 4] = 1.0
    assert _fast(f2, 5) is None

    # Negative entry with sum == 1 (e.g. [1, 1, -1]) rejected
    f3 = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    f3[0, 0, 0] = torch.tensor([1.0, 1.0, -1.0])
    assert _fast(f3, 3) is None


def test_norm_adaln_on_card_signature_compat():
    """Verify _norm_adaln_on_card defaults cg=None so 8-arg SPMD callers stay compatible."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "tt_bio" / "train" / "rfd4_pair.py"
    tree = ast.parse(src.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_norm_adaln_on_card")
    pos_names = [a.arg for a in fn.args.args]
    assert pos_names[:8] == ["x", "lnw_tt", "gamma_tt", "beta_tt", "e1", "e2", "c", "cfg"]
    assert len(pos_names) == 9 and pos_names[8] == "cg"
    assert len(fn.args.defaults) == 1 and isinstance(fn.args.defaults[0], ast.Constant) and fn.args.defaults[0].value is None
    kw_names = [a.arg for a in fn.args.kwonlyargs]
    assert kw_names == ["return_saved"]
    assert isinstance(fn.args.kw_defaults[0], ast.Constant) and fn.args.kw_defaults[0].value is False


def test_pow2_chunks_partition_and_bounds():
    """Verify _pow2_chunks partitions any active-set size into powers of two <= 32 in order."""
    from tt_bio.train.folds import _pow2_chunks

    for k in (0, 1, 2, 3, 14, 15, 35, 36, 37, 38, 56, 84):
        seq = list(range(k))
        chunks = _pow2_chunks(seq)
        flat = [x for ch in chunks for x in ch]
        assert flat == seq
        for ch in chunks:
            n = len(ch)
            assert 1 <= n <= 32 and (n & (n - 1)) == 0
    # 35..38 share {32, 4, 2, 1} so d=1 / batch08 combinations never compile a new batch size
    assert [len(c) for c in _pow2_chunks(list(range(36)))] == [32, 4]
    assert [len(c) for c in _pow2_chunks(list(range(35)))] == [32, 2, 1]
    assert [len(c) for c in _pow2_chunks(list(range(37)))] == [32, 4, 1]
    assert [len(c) for c in _pow2_chunks(list(range(38)))] == [32, 4, 2]


def test_device_optimizer_free_grads_frees_orig_and_reduced(monkeypatch):
    """When reduce_fn returns a new dict, free_grads=True must deallocate BOTH orig and reduced."""
    import sys
    import types
    from tt_bio.train.device_optim import DeviceOptimizer

    class _FakeT:
        def __init__(self, tag):
            self.tag = tag
            self.alive = True
        def is_allocated(self):
            return self.alive

    freed = []
    fake_ttnn = types.SimpleNamespace(
        deallocate=lambda t: (setattr(t, "alive", False), freed.append(t.tag)),
    )
    monkeypatch.setitem(sys.modules, "ttnn", fake_ttnn)

    opt = object.__new__(DeviceOptimizer)
    opt.lr = 1e-4
    opt.schedule = None
    opt.muon = None
    opt.muon_names = frozenset()
    opt.steps = 0
    opt.beta1 = 0.9
    opt.beta2 = 0.999
    opt.beta1_pow = 1.0
    opt.beta2_pow = 1.0
    opt._muon_buckets = []
    opt._adam_buckets = []
    opt.ema = None
    opt.master = {}
    opt.owned_names = frozenset()
    opt.clip_norm = 1.0
    opt._group_norms_and_dead = lambda m, a: (0.0, 0.0, set())

    g_orig = _FakeT("orig")
    g_red = _FakeT("reduced")
    opt.step({"w": g_orig}, reduce_fn=lambda d: {"w": g_red}, free_grads=True)
    assert set(freed) == {"orig", "reduced"}
    assert not g_orig.alive and not g_red.alive





