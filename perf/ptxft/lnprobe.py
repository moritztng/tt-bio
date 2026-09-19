"""Is ag.layer_norm shape-dependent? The single-track check says the s norm is 4.1e-01
off on a [1, S, c] input while the z norm is 2.0e-02 on [S, S, c], same op, same gain
layout. One op, one variable: the shape."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ttnn
from tt_bio.tenstorrent import get_device
from tt_bio import autograd as ag, train as ft

def rel(a, b):
    a, b = np.float64(a), np.float64(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

def main():
    dev = get_device()
    rng = np.random.default_rng(3)
    print("shape                dtype     rel L2")
    for shp in [(1, 64, 384), (64, 384), (64, 64, 256), (1, 64, 256),
                (1, 32, 384), (1, 128, 384), (4, 64, 384), (64, 64, 384)]:
        for dt, nm in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")):
            x = (rng.standard_normal(shp) * 0.5).astype(np.float32)
            g = np.ones((1, 1, shp[-1]), np.float32)
            b = np.zeros((1, 1, shp[-1]), np.float32)
            t = ag.Tensor(ft.to_device(x, dev, dtype=dt))
            gg = ag.Tensor(ft.to_device(g, dev, dtype=dt))
            bb = ag.Tensor(ft.to_device(b, dev, dtype=dt))
            out = ft.to_host(ag.layer_norm(t, gg, bb, eps=1e-5).value).reshape(shp)
            m = x.mean(-1, keepdims=True)
            v = ((x - m) ** 2).mean(-1, keepdims=True)
            ref = (x - m) / np.sqrt(v + 1e-5)
            print(f"{str(shp):<20} {nm:>5} {rel(out, ref):>10.3e}")

main()
