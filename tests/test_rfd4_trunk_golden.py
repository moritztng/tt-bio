"""RFD4-Proteina trunk modules on the card against rfd4-ref's goldens: the RELEASED flow weights and the
inputs upstream itself fed each module (`state/rfd4/goldens/INDEX.json`, feature `unconditional_128`, seed 0,
CPU fp32, upstream 2db07bb4).

Weights: `flow_trunk.pt`, the `nn.` trunk tensors of `rfd4_proteina_flow-ema.ckpt` with the PEFT LoRA on
`to_qkv` / `to_out_node` merged exactly (W = base + (alpha / r) B A, alpha 64, r 32). Release dims: 14 blocks,
token 768, 12 heads, pair 384, cond 256.

Each golden is graded three ways:
* the float64 host reference (`rfd4_reference`) on the golden's input against the golden's output, which
  checks the transcription on trained weights;
* the card against the golden;
* the card against the same chain at bf16-mixed (bf16 weights, bf16 op boundaries, fp32 residual streams),
  the floor, which is what a bf16-mixed GPU computes.
The golden's 100 tokens are bucketed to 128 (padding masked); only real tokens are graded.

Run: RFD4_GOLDENS=~/rfd4-goldens TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_rfd4_trunk_golden.py
"""
import os
from pathlib import Path

import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T
from tt_bio.rfd4 import trunk as TR

import rfd4_reference as REF

ROOT = Path(os.environ.get("RFD4_GOLDENS", Path.home() / "rfd4-goldens"))
FEATURE = ROOT / "unconditional_128"
WEIGHTS = ROOT / "trunk-weights" / "flow_trunk.pt"
HEADS = 12
BUCKET = 32
BAR = 5e-2
FLOOR_FACTOR = 3.0
REF_BAR = 1e-4  # float64 transcription vs upstream's own fp32 run

pytestmark = pytest.mark.skipif(not (FEATURE.is_dir() and WEIGHTS.is_file()),
                                reason=f"rfd4-ref goldens / merged weights not under {ROOT}")

BLOCKS = [1, 13]
PAIR_UPDATES = [0, 4, 8, 12]


@pytest.fixture(scope="module")
def weights():
    return {k: v.double() for k, v in torch.load(WEIGHTS, map_location="cpu", weights_only=False)["weights"].items()}


def _golden(name):
    d = torch.load(FEATURE / name, map_location="cpu", weights_only=False)
    return [a.double() if a.is_floating_point() else a.double() for a in d["args"]], d["output"].double()


def _pad(t, n, dims):
    for d in dims:
        pad = [0, 0] * (t.dim() - 1 - d) + [0, n - t.shape[d]]
        t = torch.nn.functional.pad(t, pad)
    return t


def _bucket(n):
    return -(-n // BUCKET) * BUCKET


def _dev(t, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rel(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def _sub(w, prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in w.items() if k.startswith(p)}


def _grade(name, got, ref64, floor, golden, m):
    g, r, f, o = (t[m] for t in (got, ref64, floor, golden))
    ref_err, err, floor_err, vs_floor, pcc = _rel(r, o), _rel(g, o), _rel(f, o), _rel(g, f), _pcc(g, o)
    print(f"\n{name}: device {err:.3e} vs golden (pcc {pcc:.6f}), bf16-mixed floor {floor_err:.3e}, "
          f"device vs floor {vs_floor:.3e}, float64 ref vs golden {ref_err:.2e}")
    assert ref_err < REF_BAR, f"{name}: the float64 reference is {ref_err:.2e} from upstream's fp32 golden"
    assert err < BAR, f"{name} {err:.3e} over the {BAR} bar"
    assert err < FLOOR_FACTOR * max(floor_err, 1e-4), f"{name} {err:.3e} over {FLOOR_FACTOR}x its floor {floor_err:.3e}"
    return err


@pytest.mark.device
@pytest.mark.parametrize("layer", BLOCKS)
def test_golden_trunk_block(weights, layer):
    dev = T.get_device()
    (x, z, cond, mask), out = _golden(f"MultiheadAttnAndTransition__nn_transformer_layers_{layer}__call0.pt")
    wl = _sub(weights, f"transformer_layers.{layer}")
    wb = {k: REF.bf16(v) for k, v in wl.items()}
    mask = mask.double()
    ref64 = REF.trunk_block(wl, x, z, cond, mask, HEADS)
    floor = REF.trunk_block(wb, x, z, cond, mask, HEADS, "", REF.bf16_mixed)
    n = _bucket(x.shape[1])
    xp, zp, cp, mp = _pad(x, n, [1]), _pad(z, n, [1, 2]), _pad(cond, n, [1]), _pad(mask, n, [1])
    mod = TR.TrunkBlock(T.WeightScope({k: v.float() for k, v in wl.items()}), TR.compute_kernel_config(), HEADS,
                        residual_dtype=ttnn.float32)
    tok, bias, _, _, res_tok, _ = TR.trunk_masks(mp, dev, residual_dtype=ttnn.float32)
    got = ttnn.to_torch(mod(_dev(xp, dev, ttnn.float32), _dev(zp, dev), _dev(cp, dev), tok, bias, res_tok)).double()
    m = mask.bool()
    _grade(f"golden block {layer}", got[:, :x.shape[1]], ref64, floor, out, m)


@pytest.mark.device
@pytest.mark.parametrize("layer", PAIR_UPDATES)
def test_golden_pair_update(weights, layer):
    dev = T.get_device()
    (x, z, mask), out = _golden(f"PairReprUpdate__nn_pair_update_layers_{layer}__call0.pt")
    p = f"pair_update_layers.{layer}"
    wl = {k: v for k, v in weights.items() if k.startswith(p + ".")}
    wb = {k: REF.bf16(v) for k, v in wl.items()}
    mask = mask.double()
    ref64 = REF.pair_update(wl, p, x, z, mask)
    floor = REF.pair_update(wb, p, x, z, mask, REF.bf16_mixed)
    n = _bucket(x.shape[1])
    xp, zp, mp = _pad(x, n, [1]), _pad(z, n, [1, 2]), _pad(mask, n, [1])
    mod = TR.PairUpdate(T.WeightScope({k[len(p) + 1:]: v.float() for k, v in wl.items()}), TR.compute_kernel_config(),
                        residual_dtype=ttnn.float32)
    tok, _, pair, pair_c, _, res_pair_c = TR.trunk_masks(mp, dev, residual_dtype=ttnn.float32)
    got = ttnn.to_torch(mod(_dev(xp, dev), _dev(zp, dev, ttnn.float32), tok, pair, pair_c, res_pair_c)).double()
    k = x.shape[1]
    pm = (mask[:, :, None] * mask[:, None, :]).bool()
    _grade(f"golden pair update {layer}", got[:, :k, :k], ref64, floor, out, pm)
