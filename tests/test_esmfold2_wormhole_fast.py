"""ESMFold2's ESMC-6B is block-fp8 on Wormhole for every loader caller, in both modes.

The bf16 ESMC-6B (~12.8 GB) does not fit a Wormhole chip's 12 GiB: a worker path (perf_regression, a
server) that loaded it in normal precision died in the allocator with TT_FATAL on the 0.15.0 gate's
Wormhole perf leg. The fold's lever set still follows the mode (fold_levers).
"""
import sys
import types

import pytest

pytest.importorskip("ttnn")


@pytest.mark.parametrize("wormhole,fast,lm_b8", [(True, False, True), (False, False, False),
                                                  (False, True, True), (True, True, True)])
def test_wormhole_lm_is_bfp8_in_both_modes(monkeypatch, wormhole, fast, lm_b8):
    from tt_bio import esmfold2_runtime as rt
    from tt_bio import tenstorrent

    class Model:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def eval(self):
            return self

    mod = types.ModuleType("modeling_esmfold2")
    mod.ESMFold2Model = Model
    monkeypatch.setitem(sys.modules, "tt_bio._vendor.esmfold2_hf.modeling_esmfold2", mod)
    monkeypatch.setattr(rt, "is_wormhole", lambda: wormhole)
    built = {}
    monkeypatch.setattr(rt, "patch_esmfold2", lambda model, **k: built.update(k) or model)
    monkeypatch.setattr(tenstorrent, "_FAST_MODE", False)
    rt.load_ttnn_esmfold2(fast=fast)
    # The LM's weight format follows the arch, the mode follows the caller; `_FAST_MODE` stays off
    # (esmfold2_runtime.fold_scope).
    assert built["lm_b8"] is lm_b8 and built["fast"] is fast
    assert tenstorrent._FAST_MODE is False


@pytest.mark.parametrize("wormhole", [True, False])
def test_mode_sets_per_arch(monkeypatch, wormhole):
    from tt_bio import esmfold2_runtime as rt, tenstorrent
    monkeypatch.delenv("TT_BIO_LEVERS", raising=False)
    monkeypatch.setattr(rt, "is_wormhole", lambda: wormhole)
    normal_set = rt.fold_levers(False).split("+")
    assert "trimul_gin_hifi2" in normal_set
    # Both archs' normal mode runs the pair SwiGLU kernel (bnwg); Wormhole's adds the LM combine.
    assert "transition_ws" in normal_set and "ws_resid" in normal_set
    assert ("ef2_lm_combine" in normal_set) is wormhole
    # Fast-only: trimul_b8in on both archs, so a Wormhole --fast fold is not the normal one; ws_lofi on Blackhole
    # only (LoFi hangs Wormhole chips).
    fast_set = rt.fold_levers(True).split("+")
    assert "ws_resid" in fast_set and fast_set.count("trimul_gin_hifi2") == 1
    assert ("ws_lofi" in fast_set) is not wormhole and "ws_lofi" not in normal_set
    assert "trimul_b8in" in fast_set and "trimul_b8in" not in normal_set
    # The T1/T2 trimul stack passed both bars on Blackhole (bt1g, bt1ng) and loses time on Wormhole.
    t12 = ("trimul_t1", "trimul_t1_c256", "trimul_t1w8", "trimul_t1_2g", "trimul_t2", "trimul_t2cm")
    assert all((n in s) is not wormhole for s in (fast_set, normal_set) for n in t12)
    # Wormhole's normal mode takes the --fast trimul pair too (wnsg), Blackhole's does not.
    normal = tenstorrent.parse_levers(rt.fold_levers(False))
    assert all((n in normal) is wormhole for n in ("trimul_t2fold", "trimul_gin_w8"))
    # The ef2_* levers ride both sets on both archs; device pair init is graded on Wormhole's fast bar only.
    for fast in (False, True):
        names = rt.fold_levers(fast).split("+")
        assert "ef2_relpos_lookup" in names and "ef2_bit_draw" in names
        assert ("ef2_pair_init" in names) is (fast or not wormhole)
    # qk_rope_heads in the LM failed ESMFold2's normal bar on Wormhole (qrn) and passed the fast one (qrf).
    # Blackhole keeps the LM as it folded before ESMC took the Wormhole LM path there (no fold graded yet).
    assert "lm_qk_chain" in normal_set and ("lm_qk_chain" in fast_set) is not wormhole
    # The legacy LM gate multiply is graded on ESMC's embeddings only; ESMFold2 keeps binary_ng.
    assert "lm_gate_ng" in normal_set and "lm_gate_ng" in fast_set
    assert "lm_sdpa_stock" in normal_set and "lm_sdpa_stock" in fast_set
    assert "lm_ln_stock" in normal_set and "lm_ln_stock" in fast_set
    assert all((n in s) is not wormhole for n in ("lm_k64", "lm_linear") for s in (normal_set, fast_set))
    # The fused token-DiT attention passed the fast bar on both archs (dsg, bqkg); its hoisted conditioning (dsq2) on
    # Wormhole only.
    assert "dit_qkvf" in fast_set and "dit_qkvf" not in normal_set
    assert ("dit_shoist" in fast_set) is wormhole and "dit_shoist" not in normal_set
    assert ("trunk_hifi2" in fast_set) is wormhole and "trunk_hifi2" not in normal_set
    assert ("ef2_atom_rope" in fast_set) is wormhole and ("ef2_atom_rope" in normal_set) is wormhole
    # trimul_t2m lost to plain T2 once trimul_bres fed the tail (wtm) and rides no set.
    assert all("trimul_t2m" not in rt.fold_levers(f).split("+") for f in (False, True))
    # trimul_bres is bit-exact in ESMFold2 folds on both archs and rides every set.
    assert all("trimul_bres" in rt.fold_levers(f).split("+") for f in (False, True))
