"""Which spelling of a row mean keeps its 1/K? bf16 and fp32, K=384 and K=128.

The metric that matters is the SYSTEMATIC part: a constant that is 2^-9 low on every row does
not average out over the 816 layer-norm backwards in a step, while a per-row rounding does.
"""
import os, sys
sys.path.insert(0, os.getcwd())
import torch, ttnn, tt_bio
from tt_bio import autograd as ag
from tt_bio.tenstorrent import get_device

_WT = os.path.realpath(os.getcwd())
assert os.path.realpath(tt_bio.__file__).startswith(_WT), tt_bio.__file__

dev = get_device()
cfg = ag.precise_config()


def variants(v, K):
    s = ttnn.sum(v, dim=-1, keepdim=True, compute_kernel_config=cfg)
    out = {
        "ttnn.mean": lambda: ttnn.mean(v, dim=-1, keepdim=True, compute_kernel_config=cfg),
        "sum * (1/K)": lambda: ttnn.multiply(s, 1.0 / K),
        "sum / K": lambda: ttnn.div(s, float(K)),
        "f32(sum) * (1/K) -> back": lambda: ttnn.typecast(
            ttnn.multiply(ttnn.typecast(s, ttnn.float32), 1.0 / K), v.dtype),
        "sum(dtype=f32) * (1/K) -> back": lambda: ttnn.typecast(
            ttnn.multiply(ttnn.sum(v, dim=-1, keepdim=True, compute_kernel_config=cfg,
                                   dtype=ttnn.float32), 1.0 / K), v.dtype),
    }
    r = {}
    for name, fn in out.items():
        try:
            r[name] = ttnn.to_torch(fn()).double()
        except Exception as e:                                            # noqa: BLE001
            r[name] = str(e)
    return r


torch.manual_seed(0)
for K, shape in ((384, (1, 384, 384)), (128, (1, 64, 64, 128))):
    for dt, tdt in (("bfloat16", ttnn.bfloat16), ("float32", ttnn.float32)):
        x = torch.randn(*shape)
        v = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=tdt,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ref = ttnn.to_torch(v).double().mean(-1, keepdim=True)
        print(f"--- K={K} {dt} ---")
        for name, got in variants(v, K).items():
            if isinstance(got, str):
                print(f"  {name:32s} ERROR {got[:90]}")
                continue
            rel = ((got - ref) / ref.abs().clamp_min(1e-30))
            relL2 = (got - ref).norm().item() / ref.norm().item()
            # the constant: a scale fit, <got,ref>/<ref,ref> - 1
            scale = ((got * ref).sum() / (ref * ref).sum()).item() - 1.0
            print(f"  {name:32s} relL2 {relL2:9.3e}  scale-1 {scale:+10.3e}  "
                  f"median|rel| {rel.abs().median().item():9.3e}")
ttnn.close_device(dev)
