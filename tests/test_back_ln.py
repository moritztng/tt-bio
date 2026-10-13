"""`back_ln.back_ln` (the trimul's last back move fused with its output layer norm) against float64
and against the route it replaces, `reblock_permute_back` into the output then `ttnn.layer_norm`.

The output's earlier channels are random rows already in place (what the earlier chunks' moves left);
the kernel moves the last chunk in, then normalizes all H channels of every row.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_back_ln.py
"""
import pytest
import torch
import ttnn

from tt_bio import back_ln as BL
from tt_bio import reblock_permute as R
from tt_bio import tenstorrent as T

EPS = 1e-5


def _dev(t, dev):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rel_rms(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("n,h,c,parts", [(64, 256, 128, None), (128, 256, 128, 2), (128, 256, 256, 1),
                                         (96, 128, 128, 4), (512, 256, 128, None)])
def test_back_ln_matches_float64_and_the_split_route(n, h, c, parts, monkeypatch):
    dev = T.get_device()
    monkeypatch.setattr(BL, "ENABLED", True)
    g = torch.Generator().manual_seed(n + h + c)
    # Einsum outputs are not centred: a per-channel offset makes the mean subtraction matter.
    chunk = (torch.randn(1, c, n, n, generator=g) * 3 + torch.randn(1, c, 1, 1, generator=g)).to(torch.bfloat16)
    prev = (torch.randn(1, n, n, h - c, generator=g) * 2 + 1).to(torch.bfloat16)
    gamma = (1 + torch.randn(h, generator=g) / 4).to(torch.bfloat16)
    beta = (torch.randn(h, generator=g) / 4).to(torch.bfloat16)

    d = torch.float64
    full = torch.cat([prev.to(d), chunk.to(d).permute(0, 2, 3, 1)], dim=-1)
    ref = torch.nn.functional.layer_norm(full, (h,), gamma.to(d), beta.to(d), EPS)

    dchunk = _dev(chunk, dev)
    dgam = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    dbet = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    # The kernel: prev's channels in place, then the fused move + LN.
    out = _dev(torch.cat([prev, torch.zeros(1, n, n, c, dtype=torch.bfloat16)], dim=-1), dev)
    assert BL.eligible(dchunk, out, h - c)
    rows = BL.weight_rows(dgam, dbet, dev)
    BL.back_ln(dchunk, out, h - c, *rows, EPS, parts=parts)
    got = ttnn.to_torch(out).to(d)

    # The route it replaces.
    out2 = _dev(torch.cat([prev, torch.zeros(1, n, n, c, dtype=torch.bfloat16)], dim=-1), dev)
    R.reblock_permute_back(dchunk, ttnn.DRAM_MEMORY_CONFIG, out=out2, c_off=h - c)
    split = ttnn.to_torch(ttnn.layer_norm(out2, weight=dgam, bias=dbet, epsilon=EPS)).to(d)

    assert torch.isfinite(got).all()
    e_got, e_split = _rel_rms(got, ref), _rel_rms(split, ref)
    print(f"n={n} h={h} c={c} parts={parts}: kernel rel_rms {e_got:.6f} max {(got - ref).abs().max():.4f} | "
          f"split route rel_rms {e_split:.6f} max {(split - ref).abs().max():.4f}")
    # One bf16 rounding of an fp32 result: ~2^-9 relative, rms well under 4e-3.
    assert e_got < 4e-3
    assert e_got <= e_split * 1.05 + 1e-4
    assert (got - ref).abs().max() < 0.1
