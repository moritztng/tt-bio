"""Where does a bf16-exact-operand fp32 matmul lose precision: K per tile, packer_l1_acc, output dtype. Float64 refs."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as tt
    dev = tt.get_device()
    up = lambda t: ttnn.from_torch(t.float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa
    g = torch.Generator().manual_seed(0)
    for l1 in (True, False):
        for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2):
            ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                                         fp32_dest_acc_en=True, packer_l1_acc=l1)
            for K in (32, 64, 384):
                x = torch.randn(1024, K, generator=g).to(torch.bfloat16).double()
                w = torch.randn(K, 256, generator=g).to(torch.bfloat16).double()
                want = x @ w
                got = ttnn.to_torch(ttnn.matmul(up(x), up(w), compute_kernel_config=ckc, dtype=ttnn.float32)).double()
                e = float((got - want).norm() / want.norm())
                print(f"l1acc={l1} {fid} K={K} bf16-exact operands rel {e:.3e}", flush=True)
    # single products: is the multiply itself exact? K=32 with one nonzero per row
    x = torch.zeros(32, 32); x[:, 0] = torch.randn(32, generator=g).to(torch.bfloat16).float()
    w = torch.zeros(32, 32); w[0, :] = torch.randn(32, generator=g).to(torch.bfloat16).float()
    ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                 fp32_dest_acc_en=True, packer_l1_acc=False)
    got = ttnn.to_torch(ttnn.matmul(up(x), up(w), compute_kernel_config=ckc, dtype=ttnn.float32)).double()
    want = x.double() @ w.double()
    print("single product rel", float((got - want).norm() / want.norm()), flush=True)


if __name__ == "__main__":
    main()
