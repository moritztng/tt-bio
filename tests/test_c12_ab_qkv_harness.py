"""Run the A/B harness's device path against a stub, so the card window is not its first run.

`perf/c12_diffusion_head/ab_qkv.py` decides whether this row's lever is real. It has never
executed: it needs a Tenstorrent chip, and the row has been dispatched `card=cpu` for four passes.
A `TypeError` or an empty `max()` on the first call would burn a board pair's worth of time, so
drive the whole thing here with a stub `ttnn` whose ops return real torch tensors.

What this pins is the harness's own logic, which is the part a device cannot check for us:
  * three arms plus the session's own A/A arm, interleaved rep by rep, cold rep discarded PER ARM;
  * torch.equal(A1, B) evaluated BEFORE any timing;
  * a SHAPE mismatch reported as a finding rather than raised, which is what a wrong destination
    would look like;
  * the three ratios and the A/A floor computed off the medians.
"""
import pathlib
import sys
import types

import pytest
import torch

OUT = pathlib.Path(__file__).resolve().parent.parent / "perf" / "c12_diffusion_head"


class _Stub(types.ModuleType):
    """Only the ttnn surface the harness touches. Tensors are real torch tensors."""

    def __init__(self, break_shape=False):
        super().__init__("ttnn")
        self.calls = []
        self._break_shape = break_shape
        self.bfloat16 = torch.bfloat16
        self.TILE_LAYOUT = "TILE"
        self.DRAM_MEMORY_CONFIG = "DRAM"

        class _MF:
            HiFi4 = "HiFi4"
        self.MathFidelity = _MF

        class _Arch:
            WORMHOLE_B0 = "wh"
            BLACKHOLE = "bh"
        self.Arch = _Arch

        def _ckc(**kw):
            return types.SimpleNamespace(**kw)
        self.types = types.SimpleNamespace(WormholeComputeKernelConfig=_ckc,
                                           BlackholeComputeKernelConfig=_ckc)
        self.experimental = types.SimpleNamespace(
            minimal_matmul=self._minimal_matmul,
            nlp_create_qkv_heads=self._create_heads)

    # --- the ops ---------------------------------------------------------------------------
    def from_torch(self, t, **kw):
        return t

    def to_torch(self, t):
        return t

    def deallocate(self, t):
        pass

    def synchronize_device(self, d):
        pass

    def reshape(self, t, dims, *a, **k):
        return t.reshape(*[d for d in dims])

    def squeeze(self, t, dim):
        return t.squeeze(dim)

    def unsqueeze(self, t, dim):
        return t.unsqueeze(dim)

    def pad(self, t, spec, value):
        pads = []
        for lo, hi in reversed(spec):
            pads += [int(lo), int(hi)]
        return torch.nn.functional.pad(t.float(), pads, value=float(value)).to(t.dtype)

    def linear(self, x, w, bias=None, **kw):
        self.calls.append("linear")
        out = x.float() @ w.float()
        if bias is not None:
            out = out + bias.float()
        return out.to(x.dtype)

    def _minimal_matmul(self, input_tensor=None, weight_tensor=None, bias_tensor=None, **kw):
        self.calls.append("minimal_matmul")
        out = input_tensor.float() @ weight_tensor.float()
        if bias_tensor is not None:
            out = out + bias_tensor.float()
        return out.to(input_tensor.dtype)

    def _create_heads(self, q, kv=None, num_heads=1, num_kv_heads=1, transpose_k_heads=False,
                      **kwargs):
        """The real op's layout: channels run (q|k|v, head, dim), and heads move to dim 1."""
        self.calls.append("nlp_create_qkv_heads")
        if kv is None:
            b, _one, s, w = q.shape
            d = w // (3 * num_heads)
            x = q.reshape(b, s, 3, num_heads, d).permute(2, 0, 3, 1, 4).contiguous()
            return x[0], x[1], x[2]
        b, _one, s, w = q.shape
        qq = q.reshape(b, s, num_heads, w // num_heads).permute(0, 2, 1, 3).contiguous()
        kvv = kv.reshape(b, kv.shape[2], 2, num_kv_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        return qq, kvv[0], kvv[1]


def _install(monkeypatch, stub):
    """The stub replaces ttnn only AFTER tt_bio has bound the real one at import.

    `mm_generic` reads `ttnn.NOC` at module scope and `tenstorrent` reads far more, so a stub
    installed first makes the import fail. The harness does `import ttnn` INSIDE its functions,
    which is what lets the swap reach it.
    """
    import tt_bio.tenstorrent            # noqa: F401  (binds the real ttnn)
    import tt_bio.triatt_qkv             # noqa: F401
    monkeypatch.setitem(sys.modules, "ttnn", stub)
    monkeypatch.syspath_prepend(str(OUT))
    sys.modules.pop("ab_qkv", None)
    import ab_qkv
    return ab_qkv


@pytest.fixture
def harness(monkeypatch):
    stub = _Stub()
    mod = _install(monkeypatch, stub)
    # head-major arm: return exactly what A1 returns, so equality is the identity case
    import tt_bio.triatt_qkv as TQ

    def fake_qkv_heads(x, w, ckc, n_heads, head_dim, dtype, cfg, bias=None, **kw):
        out = x.float() @ w.float()
        if bias is not None:
            out = out + bias.float()
        out = out.to(x.dtype)
        b, s, _ = out.shape
        return tuple(out.reshape(b, s, 3, n_heads, head_dim)
                        .permute(2, 0, 3, 1, 4).contiguous()[i] for i in range(3))

    def fake_atom(s, w_q, b_q, s_kv, w_kv, ckc, n_heads, head_dim, dtype, cq, ck, **kw):
        q = (s.float() @ w_q.float() + (b_q.float() if b_q is not None else 0)).to(s.dtype)
        kv = (s_kv.float() @ w_kv.float()).to(s.dtype)
        B, K, W, _ = s.shape
        qh = q.reshape(B, K, W, n_heads, head_dim).permute(0, 1, 3, 2, 4) \
              .reshape(B, K * n_heads, W, head_dim).contiguous()
        A = s_kv.shape[2]
        kvh = kv.reshape(B, K, A, 2, n_heads, head_dim).permute(3, 0, 1, 4, 2, 5) \
                .reshape(2, B, K * n_heads, A, head_dim).contiguous()
        return qh, kvh[0], kvh[1]

    monkeypatch.setattr(TQ, "qkv_heads", fake_qkv_heads)
    monkeypatch.setattr(TQ, "atom_qkv_heads", fake_atom)
    # widths whose (kt, nt) keys really are in _MM_BLOCK, so `_qkv_mm_config` returns a config:
    # fused qkv [128, 3*4*32] -> (4, 12); atom q [128, 4*32] -> (4, 4); atom kv [128, 2*4*32] -> (4, 8).
    # seq 128 so mt = 4 clears the (4, 12) entry`s M_block of 4; at 64 the config gate declines.
    monkeypatch.setattr(mod, "SIGS", [("dit_token", 1, 128, 128, 4, 32, 32, 4800, 0.20470)])
    monkeypatch.setattr(mod, "ATOM", ("atom_block", 1, 4, 32, 64, 128, 4, 32, 1200, 0.09016))
    return mod, stub


DEV = types.SimpleNamespace(arch=lambda: "bh")


def test_compare_reports_a_shape_mismatch_instead_of_raising(harness):
    mod, _stub = harness
    a = [torch.zeros(2, 3)]
    assert mod._compare(a, [torch.zeros(2, 3)])["torch_equal_A1_vs_B"] is True
    bad = mod._compare(a, [torch.zeros(3, 2)])
    assert bad["torch_equal_A1_vs_B"] is False
    assert bad["max_abs_A1_vs_B"] is None
    assert bad["shape_mismatch"] == {"A1": [[2, 3]], "B": [[3, 2]]}
    # and an empty comparison must not blow up on max()
    assert mod._compare([], [])["max_abs_A1_vs_B"] is None


def test_fused_arms_run_interleaved_and_report_the_ratios(harness):
    mod, stub = harness
    reps = 3
    r = mod._arms(DEV, mod.SIGS[0], reps)
    assert r["torch_equal_A1_vs_B"] is True, r
    assert r["shape_mismatch"] is None
    assert set(r["ms_per_call"]) == {"A0_linear_split", "A1_mm_split", "B_head_major",
                                     "AA_control"}
    for arm, v in r["raw_ms"].items():
        assert len(v) == reps, f"{arm}: cold rep not discarded per arm ({len(v)} kept)"
    assert r["B_over_A1"] > 0 and r["A1_over_A0"] > 0 and r["aa_floor_pct"] >= 0
    assert "minimal_matmul" in stub.calls and "linear" in stub.calls


def test_atom_arms_run_and_the_split_is_the_shipped_chain(harness):
    mod, stub = harness
    stub.calls.clear()
    r = mod._atom_arms(DEV, 2)
    assert r["torch_equal_A1_vs_B"] is True, r
    assert r["sig"] == "atom_block"
    assert all(len(v) == 2 for v in r["raw_ms"].values())
    # the A0/A1 arms must go through the pad the head-major arm deletes
    assert "nlp_create_qkv_heads" in stub.calls


def test_a_wrong_destination_is_a_finding_not_a_crash(harness, monkeypatch):
    """If the head-major arm returned the wrong shape, the harness must SAY so and keep going."""
    mod, _stub = harness
    import tt_bio.triatt_qkv as TQ
    good = TQ.qkv_heads
    monkeypatch.setattr(TQ, "qkv_heads",
                        lambda *a, **k: tuple(t.transpose(-1, -2).contiguous()
                                              for t in good(*a, **k)))
    r = mod._arms(DEV, mod.SIGS[0], 2)
    assert r["torch_equal_A1_vs_B"] is False
    assert r["shape_mismatch"] is not None
    assert r["ms_per_call"], "timing must still have run, so the session is not wasted"
