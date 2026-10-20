"""metal's gelu on the device against float64: the wheel's accurate path, its approx path (calculate_gelu_f32 under the
gelu_f32 overlay that tt_bio puts in place at import) and the same two fused into a matmul's pack (fc1's use).

    python perf/spd_m_saprot/gelu_check.py --out gelu.jsonl
"""
import argparse
import json

import torch
import ttnn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    bits = torch.arange(0, 1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    grid = bits[torch.isfinite(bits) & (bits.float().abs() < 8)]
    n = grid.numel() // 1024 * 1024
    xs = {"bf16_grid": grid[:n].reshape(-1, 1024), "normal_1.5": (torch.randn(256, 1024, generator=torch.Generator().manual_seed(0)) * 1.5).to(torch.bfloat16)}
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True, packer_l1_acc=False)
    out = open(a.out, "w")
    for name, x in xs.items():
        ref = torch.nn.functional.gelu(x.double())
        xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        eye = ttnn.from_torch(torch.eye(1024, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        for approx in (False, True):
            arms = {"unary": lambda: ttnn.gelu(ttnn.typecast(xd, ttnn.float32), fast_and_approximate_mode=approx),
                    "matmul": lambda: ttnn.linear(xd, eye, activation="gelu_approx" if approx else "gelu",
                                                  compute_kernel_config=ck, dtype=ttnn.float32)}
            for arm, fn in arms.items():
                got = ttnn.to_torch(fn()).double()
                err = (got - ref).abs()
                rec = dict(x=name, approx=approx, arm=arm, max_abs=err.max().item(), at=x.flatten()[err.argmax()].item(),
                           rel_l2=((got - ref).norm() / ref.norm()).item())
                print(json.dumps(rec), flush=True)
                out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
