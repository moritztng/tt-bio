"""Smoke test: Fp32Pairformer (device, fp32) vs reference PairformerModule (host, fp32).

Random weights, 2 blocks, L=192 padded / 141 real. Verifies shapes, mask
handling, and numerics before wiring into the real affinity model. Also runs
the full Fp32PairformerModule wrapper (padding + mask cache) at L=141.
"""

import torch
import ttnn
from ttnn.device import Arch

import tt_bio.tenstorrent as T
from tt_bio.reference import PairformerModule as RefPairformer

L_REAL = 141
L_PAD = 192
N_BLOCKS = 2

torch.manual_seed(0)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def make_ckc(device):
    kernel_cls = (
        ttnn.types.WormholeComputeKernelConfig
        if device.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig
    )
    return kernel_cls(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def main() -> None:
    device = T.get_device()
    if True:
        ref = RefPairformer(
            384, 128, N_BLOCKS,
            num_heads=16, dropout=0.0,
            pairwise_head_width=32, pairwise_num_heads=4,
            v2=True,
        )
        ref.eval()
        # final_init_ zeroes every output projection (p_out, proj_o, linear_o,
        # fc3), which makes each residual branch the identity and a device-vs-
        # reference comparison vacuously bit-exact. Re-randomize all 2D weights
        # so the test actually exercises the math.
        with torch.no_grad():
            for p in ref.parameters():
                if p.ndim == 2:
                    p.copy_(torch.randn_like(p) * 0.1)
        sd = ref.state_dict()

        ckc = make_ckc(device)
        mod = T.Fp32Pairformer(2, 32, 4, 24, 16, True, sd, ckc)

        s = torch.randn(1, L_PAD, 384)
        z = torch.randn(1, L_PAD, L_PAD, 128)
        mask = torch.ones(1, L_PAD)
        mask[0, L_REAL:] = 0
        pair_mask = mask[:, :, None] * mask[:, None, :]

        ref64 = RefPairformer(
            384, 128, N_BLOCKS,
            num_heads=16, dropout=0.0,
            pairwise_head_width=32, pairwise_num_heads=4,
            v2=True,
        ).double()
        ref64.load_state_dict({k: v.double() for k, v in sd.items()})
        ref64.eval()

        with torch.no_grad():
            s_ref, z_ref = ref(s, z, mask, pair_mask)
            s_64, z_64 = ref64(s.double(), z.double(), mask.double(), pair_mask.double())

        def tt(x: torch.Tensor) -> ttnn.Tensor:
            return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)

        s_tt = tt(s)
        z_tt = tt(z)
        mask_tt = tt(pair_mask.unsqueeze(-1))
        tri_mask_tt = tt(((pair_mask - 1.0) * 1e9).reshape(L_PAD, 1, 1, L_PAD))
        s_mask_tt = tt((1 - mask).reshape(1, 1, 1, -1) * -1e6)

        s_out, z_out = mod(s_tt, z_tt, mask_tt, tri_mask_tt, s_mask_tt)
        s_dev = ttnn.to_torch(s_out)
        z_dev = ttnn.to_torch(z_out)

        sl_s = (slice(None), slice(0, L_REAL), slice(None))
        sl_z = (slice(None), slice(0, L_REAL), slice(0, L_REAL), slice(None))
        print(f"[direct module, {N_BLOCKS} blocks, L={L_PAD} pad / {L_REAL} real]")
        print(f"  scale: s absmax={s_ref.abs().max():.3e}  z absmax={z_ref.abs().max():.3e}")
        print(
            f"  s: pcc={pcc(s_dev[sl_s], s_ref[sl_s]):.6f} "
            f"maxabs={(s_dev[sl_s] - s_ref[sl_s]).abs().max():.3e}"
        )
        print(
            f"  z: pcc={pcc(z_dev[sl_z], z_ref[sl_z]):.6f} "
            f"maxabs={(z_dev[sl_z] - z_ref[sl_z]).abs().max():.3e}"
        )
        # fp64 arbitration: is the device fp32 further from truth than host fp32?
        print(
            f"  vs fp64  s: device maxabs={(s_dev[sl_s].double() - s_64[sl_s]).abs().max():.3e} "
            f"hostfp32 maxabs={(s_ref[sl_s].double() - s_64[sl_s]).abs().max():.3e}"
        )
        print(
            f"  vs fp64  z: device maxabs={(z_dev[sl_z].double() - z_64[sl_z]).abs().max():.3e} "
            f"hostfp32 maxabs={(z_ref[sl_z].double() - z_64[sl_z]).abs().max():.3e}"
        )

        # Full wrapper path: unpadded L=141 input, exercises padding + mask cache.
        wrapper = T.Fp32PairformerModule(N_BLOCKS, 32, 4, 24, 16, True)
        wrapper.load_state_dict(sd, strict=False)
        s_in = torch.randn(1, L_REAL, 384)
        z_in = torch.randn(1, L_REAL, L_REAL, 128)
        mask_in = torch.ones(1, L_REAL)
        pair_mask_in = mask_in[:, :, None] * mask_in[:, None, :]
        with torch.no_grad():
            s_ref2, z_ref2 = ref(s_in, z_in, mask_in, pair_mask_in)
        s_dev2, z_dev2 = wrapper(s_in, z_in, mask_in, pair_mask_in)
        print(f"[wrapper, {N_BLOCKS} blocks, L={L_REAL} unpadded]")
        print(
            f"  s: pcc={pcc(s_dev2, s_ref2):.6f} "
            f"maxabs={(s_dev2 - s_ref2).abs().max():.3e}"
        )
        print(
            f"  z: pcc={pcc(z_dev2, z_ref2):.6f} "
            f"maxabs={(z_dev2 - z_ref2).abs().max():.3e}"
        )
        # second call exercises the mask cache path
        s_dev3, z_dev3 = wrapper(s_in, z_in, mask_in, pair_mask_in)
        print(
            f"  z (cached-mask rerun): pcc={pcc(z_dev3, z_ref2):.6f} "
            f"maxabs={(z_dev3 - z_ref2).abs().max():.3e}"
        )


if __name__ == "__main__":
    main()
