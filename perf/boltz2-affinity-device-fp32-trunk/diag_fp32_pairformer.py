"""Diagnose where the device-fp32 vs host-fp32 z error concentrates:
per-block error growth (random walk vs localized defect) and outlier positions.
"""

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
    s, z, mask, pair_mask = io["s"], io["z"], io["mask"], io["pair_mask"]
    L = z.shape[1]

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd_all = ckpt.get("state_dict", ckpt)
    sd = {k[len("pairformer_module."):]: v.float()
          for k, v in sd_all.items() if k.startswith("pairformer_module.")}

    device = T.get_device()

    for nblocks in (1, 4, 16, 64):
        ref = RefPairformer(384, 128, nblocks, num_heads=16, dropout=0.0, v2=True)
        sub = {k: v for k, v in sd.items()
               if not k.startswith("layers.") or int(k.split(".")[1]) < nblocks}
        ref.load_state_dict(sub)
        ref.eval()
        with torch.no_grad():
            s_ref, z_ref = ref(s, z, mask, pair_mask)

        wrapper = T.Fp32PairformerModule(nblocks, 32, 4, 24, 16, True)
        wrapper.load_state_dict(sub, strict=False)
        with torch.no_grad():
            s_dev, z_dev = wrapper(s, z, mask, pair_mask)
        dz = (z_dev - z_ref).abs()
        print(f"blocks={nblocks:3d}  z pcc={pcc(z_dev, z_ref):.7f} "
              f"relmax={(dz.max() / z_ref.abs().max()):.3e} meanabs={dz.mean():.3e}",
              flush=True)
        if nblocks == 64:
            # where are the worst z errors? [i, j] positions and channels
            dzi = dz[0].amax(dim=(1, 2))  # per row i
            top_i = dzi.topk(5)
            print("  worst rows i:", [int(i) for i in top_i.indices],
                  [f"{float(v):.2e}" for v in top_i.values])
            dzc = dz[0].amax(dim=(0, 1))  # per channel
            top_c = dzc.topk(5)
            print("  worst channels:", [int(c) for c in top_c.indices],
                  [f"{float(v):.2e}" for v in top_c.values])
            flat = dz[0].flatten()
            thr = torch.quantile(flat, 0.99999)
            print(f"  z abs err: p50={flat.median():.2e} p99999={thr:.2e} "
                  f"max={flat.max():.2e}")


if __name__ == "__main__":
    main()
