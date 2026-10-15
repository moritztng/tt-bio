"""addcmul_at against float64: column windows of wide operands, bf16 (FPU) and fp32 (SFPU) inputs,
bf16 / fp32 / bfp8 outputs, at the DiT's shapes (512 rows, 768-column windows of a 36864-wide tensor).

Prints one line per case: rel_rms and max |err| vs float64, and the same for ttnn.addcmul on the sliced
operands at the same formats (the incumbent), so the kernel is judged against what it replaces.
usage: python perf/spd_difflin/aca_check.py   (TT_VISIBLE_DEVICES pins the chip)
"""
import sys

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio import addcmul_at as A  # noqa: E402


def err(got, ref):
    d = got.double() - ref
    return (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(), d.abs().max().item()


def main():
    dev = T.get_device()
    torch.manual_seed(0)
    R, W, NW = 512, 768, 48
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    bad = 0
    for in_dt in (ttnn.bfloat16, ttnn.float32):
        xw = torch.randn(R, W * NW)
        zw = torch.sigmoid(torch.randn(R, W * NW))
        y = torch.randn(R, W)
        xd, zd, yd = up(xw, in_dt), up(zw, in_dt), up(y, in_dt)
        # the reference sees the operands as the device holds them
        xh, zh, yh = (ttnn.to_torch(t).double() for t in (xd, zd, yd))
        for out_dt in (None, ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b):
            for j in (0, 1, 47):
                off = j * W
                got = A.addcmul_at(xd, yd, zd, W, x_off=off, z_off=off, out_dtype=out_dt)
                if got is None:
                    print(f"in {in_dt} out {out_dt} win {j}: DECLINED", flush=True)
                    bad += 1
                    continue
                ref = xh[:, off:off + W] + yh * zh[:, off:off + W]
                e = err(ttnn.to_torch(got), ref)
                xs, zs = xd[:, off:off + W], zd[:, off:off + W]
                inc = ttnn.addcmul(xs, yd, zs, value=1.0)
                e0 = err(ttnn.to_torch(inc), ref)
                ok = e[0] <= max(2 * e0[0], 1e-2 if out_dt == ttnn.bfloat8_b else 1e-6)
                bad += not ok
                print(f"in {in_dt} out {out_dt} win {j}: rel_rms {e[0]:.3e} max {e[1]:.3e} | "
                      f"ttnn.addcmul {e0[0]:.3e} {e0[1]:.3e} {'ok' if ok else 'BAD'}", flush=True)
                for t in (got, inc, xs, zs):
                    ttnn.deallocate(t)
    print("STATS", A.STATS, "FAIL" if bad else "PASS", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
