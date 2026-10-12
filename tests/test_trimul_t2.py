"""The trimul output tail T2 (`trimul_t2`) against the incumbent ops and float64, both forms.

Plain form: `z += (layer_norm(x) @ Wp) * sigmoid(g)` with g the fused in-projection's g_out.
GATE form (`wg=`, lever `trimul_t2g`): g is the trimul's input norm and T2 computes `sigmoid(g @ Wg)` itself, the
shape ESMFold2 calls under trimul_gin. c_z 128 is the OpenFold3 / Boltz-2 key, 256 ESMFold2's.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_trimul_t2.py
"""
import pytest
import torch
import ttnn

from tt_bio import mm_generic as MG
from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TTL
from tt_bio.af2 import compute_kernel_config


def _dev(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rel_rms(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("c", [128, 256])
def test_t2_matches_float64_like_the_incumbent(c, gate, monkeypatch):
    monkeypatch.setattr(TTL, "T2", True)
    dev = T.get_device()
    n = 128     # a size the bench also ran; the model path declines below MIN_N on its own
    gen = torch.Generator().manual_seed(c + gate)
    h = {"x": torch.randn(1, n, n, c, generator=gen) * 2 + torch.randn(1, n, n, 1, generator=gen),
         "g": torch.randn(1, n, n, c, generator=gen), "z": torch.randn(1, n, n, c, generator=gen),
         "w": torch.randn(c, c, generator=gen) / c ** 0.5, "wg": torch.randn(c, c, generator=gen) / c ** 0.5,
         "gamma": 1 + 0.1 * torch.randn(c, generator=gen), "beta": 0.1 * torch.randn(c, generator=gen)}
    h = {k: v.to(torch.bfloat16) for k, v in h.items()}
    f = {k: v.double() for k, v in h.items()}
    mu = f["x"].mean(-1, keepdim=True)
    xn = (f["x"] - mu) / torch.sqrt((f["x"] - mu).pow(2).mean(-1, keepdim=True) + 1e-5) * f["gamma"] + f["beta"]
    ref = f["z"] + (xn @ f["w"]) * torch.sigmoid(f["g"] @ f["wg"] if gate else f["g"])

    d = {k: _dev(v, dev) for k, v in h.items()}
    kcfg = T.trunk_compute_kernel_config(compute_kernel_config())
    wg = d["wg"] if gate else None

    xn_d = ttnn.layer_norm(d["x"], weight=d["gamma"], bias=d["beta"], epsilon=1e-5,
                           compute_kernel_config=T.ln_compute_kernel_config(kcfg))
    p = T._trimul_out_proj(xn_d, d["w"], kcfg, None)
    g = T._trimul_out_proj(d["g"], wg, kcfg, None) if gate else d["g"]
    p = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    inc = ttnn.to_torch(ttnn.add(d["z"], p)).double().reshape(ref.shape)

    z = _dev(h["z"], dev)
    y = TTL.fused_tail_ln(d["x"], d["gamma"], d["beta"], d["w"], d["g"], z, MG.ckc_args(kcfg),
                          tuple(T.COMPUTE_GRID_MAIN), wg=wg)
    assert y is not None, TTL.REJECTS
    out = ttnn.to_torch(y).double().reshape(ref.shape)
    assert torch.isfinite(out).all()
    e_inc, e_t2 = _rel_rms(inc, ref), _rel_rms(out, ref)
    assert e_inc < 5e-3, e_inc
    # bench (perf/spd_trimul/tail_t2.py, n 512): GATE within 1.12x of the incumbent, plain 1.45x
    assert e_t2 <= (1.25 if gate else 1.6) * e_inc, (e_t2, e_inc)
