"""k1_linear with M cut into blocks: bit-identical to one block, and compiles at 1024-token sizes.

Wormhole, one chip. Run with TT_VISIBLE_DEVICES set:  python perf/spd/k1_block_probe.py
"""
import torch
import ttnn

from tt_bio import tenstorrent as T

dev = T.get_device()
torch.manual_seed(0)
w = torch.randn(128, 128) / 11.3
b = torch.randn(1, 128)
tt = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=False)
wt, bt = tt(w), tt(b)


def run(x, budget):
    T._K1_CB_BUDGET = budget
    T._k1_program.cache_clear()
    y = T.k1_linear(tt(x), wt, bt, compute_kernel_config=ckc, dtype=ttnn.float32)
    return ttnn.to_torch(y).float()


ok = True
for rows in (5 * 5919, 5 * 8448, 5 * 9216):
    x = torch.randn(1, rows, 128)
    ref = (x.double() @ w.double() + b.double()).float()
    whole = None
    try:
        whole = run(x, 1 << 40)  # one block per core, the program before the cut
    except RuntimeError as e:
        print(f"rows {rows}: one block does not compile ({str(e).splitlines()[0][:90]})")
    cut = run(x, 1_400_000)
    small = run(x, 200_000)
    err = ((cut - ref).abs().max() / ref.abs().max()).item()
    same = [torch.equal(cut, small)] + ([torch.equal(cut, whole)] if whole is not None else [])
    print(f"rows {rows}: max|cut - f64| / max|f64| {err:.2e}  bit-identical across blockings {all(same)}")
    ok &= all(same) and torch.isfinite(cut).all().item() and err < 1e-2
print("PROBE", "PASS" if ok else "FAIL")
