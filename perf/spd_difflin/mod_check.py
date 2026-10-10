"""Float64 error and device time of the adaLN modulate as one addcmul instead of multiply + add.

    TT_VISIBLE_DEVICES=<chip> TT_BIO_LEVERS=normal python perf/spd_difflin/mod_check.py

AdaLN today writes `layer_norm(a) * sigmoid(s_scale)` (the sigmoid as multiply's b activation) and then
adds `s_bias` in a second pass, so the [M, rows, C] activation is read twice and written twice. One
`ttnn.addcmul(s_bias, ln, sigmoid(s_scale))` reads it once and writes it once; the sigmoid moves onto
the sample-invariant [1, rows, C] scale. Each arm prints whether it equals today's output bit for bit,
its error against float64, and its time (the fused arm's time includes its sigmoid).
"""
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

SHAPES = {"dit": (5, 768, 768), "atom": (5, 5920, 128)}


def err(y, r):
    e = y - r
    return f"rms {e.pow(2).mean().sqrt():.3e} max {e.abs().max():.3e}"


def timed(fn, n=20):
    dev = T.get_device()
    ttnn.deallocate(fn()); ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / n * 1e6


def main():
    dev = T.get_device()
    print("arch", T.arch_name(), flush=True)
    torch.manual_seed(11)
    up = lambda x: ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    for name, (M, R, C) in SHAPES.items():
        a, sc, sb = torch.randn(M, R, C), torch.randn(1, R, C), torch.randn(1, R, C)
        ref = a.double() * torch.sigmoid(sc.double()) + sb.double()
        ta, tsc, tsb = up(a), up(sc), up(sb)

        def cur(dt):
            x = ttnn.multiply(ta, tsc, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            y = ttnn.add(x, tsb, dtype=dt)
            ttnn.deallocate(x)
            return y

        def fused(dt, order):
            sg = ttnn.sigmoid(tsc)
            # addcmul takes no dtype: a bf16 output goes through a preallocated output_tensor.
            out = {} if dt == ta.dtype else {"output_tensor": ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(ta.shape)), dt, ttnn.TILE_LAYOUT, ta.device())}
            if order == 0:
                y = ttnn.addcmul(tsb, ta, sg, value=1.0, **out)
            elif order == 1:
                y = ttnn.addcmul(tsb, sg, ta, value=1.0, **out)
            else:  # mac(a, b, c) = a * b + c, the full-size operand first (fp32 only)
                y = ttnn.mac(ta, sg, tsb) if not out else ttnn.typecast(ttnn.mac(ta, sg, tsb), dt)
            ttnn.deallocate(sg)
            return y

        for dl, dt in (("f32", ttnn.float32), ("bf16", ttnn.bfloat16)):
            yc = ttnn.to_torch(cur(dt)).float()
            print(f"{name} out={dl} cur   vs f64 {err(yc.double(), ref)} {timed(lambda: cur(dt)):8.1f} us",
                  flush=True)
            for order in (0, 1, 2):
                try:
                    yf = ttnn.to_torch(fused(dt, order)).float()
                except Exception as e:  # an order ttnn cannot broadcast
                    print(f"{name} out={dl} fused{order} ERR {' | '.join(l for l in str(e).splitlines() if l.strip())[:400]}", flush=True)
                    continue
                print(f"{name} out={dl} fused{order} equal_cur {torch.equal(yf, yc)} "
                      f"wrong {(yf != yc).sum().item()} vs f64 {err(yf.double(), ref)} "
                      f"{timed(lambda: fused(dt, order)):8.1f} us", flush=True)
            sg_us = timed(lambda: ttnn.sigmoid(tsc))
            print(f"{name} sigmoid of the [1, {R}, {C}] scale alone {sg_us:.1f} us", flush=True)


if __name__ == "__main__":
    main()
