"""Isolate the real-weight pair-track parity: run each pair-track op of block 0 of
finetuning_ptm_1.pt individually (device vs reference, real weights, same synthetic
input). Tells whether one op is bf16-sensitive at real magnitudes vs error accumulation
across the 6 sequential pair-track residuals."""
import torch
import ttnn

from tt_bio._vendor.openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming)
from tt_bio._vendor.openfold.model.triangular_attention import (
    TriangleAttentionStartingNode, TriangleAttentionEndingNode)
from tt_bio._vendor.openfold.model.outer_product_mean import OuterProductMean
from tt_bio._vendor.openfold.model.pair_transition import PairTransition
from tt_bio.protenix_weights import remap_triangle_multiplication, remap_outer_product_mean
from tt_bio.tenstorrent import (get_device, TriangleMultiplication, TriangleAttention,
                                OuterProductMean as TTOpm)
from tt_bio.openfold import ReluTransition

CKPT = "/home/ttuser/openfold_ckpt/finetuning_ptm_1.pt"
C_M, C_Z, C_HID_MUL, C_HID_PAIR = 256, 128, 128, 32
H_PAIR, N_SEQ, N_RES = 4, 8, 64


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _sc(sd, p):
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def main(seed=0):
    torch.manual_seed(seed)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    b0 = {k[len("evoformer.blocks.0."):]: v for k, v in sd.items() if k.startswith("evoformer.blocks.0.")}
    ta = lambda d: {k.replace("mha.", ""): v for k, v in d.items()}
    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = torch.randn(1, N_RES, N_RES, C_Z) * 0.5
    m = torch.randn(1, N_SEQ, N_RES, C_M) * 0.5

    def run_ref(mod, pfx, x):
        mod.load_state_dict(_sc(b0, pfx), strict=False)
        mod.eval()
        with torch.no_grad():
            return mod(x)

    cases = []
    # tri_mul_out / in
    for name, cls, ending in [("tri_mul_out", TriangleMultiplicationOutgoing, False),
                              ("tri_mul_in", TriangleMultiplicationIncoming, True)]:
        ref = run_ref(cls(C_Z, C_HID_MUL), f"core.{name}.", z.clone())
        ttb = TriangleMultiplication(ending, remap_triangle_multiplication(_sc(b0, f"core.{name}.")), cfg)
        out = torch.Tensor(ttnn.to_torch(ttb(ft(z)))).float().reshape(ref.shape)
        cases.append((name, _pcc(out, ref)))
    # tri_att_start / end
    for name, cls, ending in [("tri_att_start", TriangleAttentionStartingNode, False),
                              ("tri_att_end", TriangleAttentionEndingNode, True)]:
        ref = run_ref(cls(C_Z, C_HID_PAIR, H_PAIR), f"core.{name}.", z.clone())
        ttb = TriangleAttention(C_HID_PAIR, H_PAIR, ending, ta(_sc(b0, f"core.{name}.")), cfg)
        out = torch.Tensor(ttnn.to_torch(ttb(ft(z)))).float().reshape(ref.shape)
        cases.append((name, _pcc(out, ref)))
    # pair_transition
    ref = run_ref(PairTransition(C_Z, 4), "core.pair_transition.", z.clone())
    ttb = ReluTransition(_sc(b0, "core.pair_transition."), cfg)
    out = torch.Tensor(ttnn.to_torch(ttb(ft(z)))).float().reshape(ref.shape)
    cases.append(("pair_transition", _pcc(out, ref)))
    # outer_product_mean (input m)
    opm = OuterProductMean(C_M, C_Z, 32); opm.load_state_dict(_sc(b0, "core.outer_product_mean."), strict=False); opm.eval()
    with torch.no_grad():
        ref = opm(m.clone())
    ttb = TTOpm(remap_outer_product_mean(_sc(b0, "core.outer_product_mean.")), cfg)
    out = torch.Tensor(ttnn.to_torch(ttb(ft(m), msa_mask=None, n_msa=N_SEQ))).float().reshape(ref.shape)
    cases.append(("outer_product_mean", _pcc(out, ref)))

    for name, p in cases:
        flag = "" if p > 0.98 else "  <-- below 0.98"
        print(f"[real op | {name:20s}] PCC={p:.5f}{flag}")


if __name__ == "__main__":
    main()
