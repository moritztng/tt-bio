"""Component verify: Fp32PairformerModule (device fp32) vs the host reference
PairformerModule (fp32 CPU) on REAL trunk I/O captured from a live affinity
predict (BOLTZ2_DUMP_TRUNK_IO) with the real boltz2_aff.ckpt weights, 64 blocks.

Bar from the workscreen: z PCC >= 0.9999 after 64 blocks.
"""

import sys
import torch
import ttnn

import tt_bio.tenstorrent as T
from tt_bio.reference import PairformerModule as RefPairformer

CKPT = "/home/ttuser/.boltz/boltz2_aff.ckpt"
IO = "/tmp/fp32trunk_in/trunk_io_fkg.pt"


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> None:
    io = torch.load(IO, weights_only=True)
    s, z = io["s"], io["z"]
    mask, pair_mask = io["mask"], io["pair_mask"]
    L = z.shape[1]
    print(f"real I/O: s {tuple(s.shape)} z {tuple(z.shape)} L={L}", flush=True)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd_all = ckpt.get("state_dict", ckpt)
    sd = {}
    for k, v in sd_all.items():
        if k.startswith("pairformer_module."):
            sd[k[len("pairformer_module."):]] = v.float()
    n_layers = len({k.split(".")[1] for k in sd if k.startswith("layers.")})
    print(f"checkpoint: {len(sd)} pairformer tensors, {n_layers} layers", flush=True)

    ref = RefPairformer(384, 128, 64, num_heads=16, dropout=0.0, v2=True)
    ref.load_state_dict(sd)
    ref.eval()

    with torch.no_grad():
        s_ref, z_ref = ref(s, z, mask, pair_mask)
    print(f"host reference done: z absmax={z_ref.abs().max():.3e}", flush=True)

    device = T.get_device()
    wrapper = T.Fp32PairformerModule(64, 32, 4, 24, 16, True)
    wrapper.load_state_dict(sd, strict=False)
    with torch.no_grad():
        s_dev, z_dev = wrapper(s, z, mask, pair_mask)

    dz = (z_dev - z_ref).abs()
    ds = (s_dev - s_ref).abs()
    print(f"[64 blocks, real I/O, L={L}]")
    print(f"  z: pcc={pcc(z_dev, z_ref):.6f} maxabs={dz.max():.3e} "
          f"rel={dz.max() / z_ref.abs().max():.3e} meanabs={dz.mean():.3e}")
    print(f"  s: pcc={pcc(s_dev, s_ref):.6f} maxabs={ds.max():.3e} "
          f"rel={ds.max() / s_ref.abs().max():.3e} meanabs={ds.mean():.3e}")
    ok = pcc(z_dev, z_ref) >= 0.9999
    print(f"  BAR: z PCC >= 0.9999 -> {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
