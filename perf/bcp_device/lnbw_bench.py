#!/usr/bin/env python3
"""The layer-norm backward at the round's shapes: arms timed alternately, graded against float64.

Arms: `composed` is `autograd._layer_norm_bw` as shipped (x-only: BindCraft 2 differentiates the
sequence, never the weights). `fused` is `tt_bio.lnbw.layer_norm_bw` when it exists. Each arm's dx is
graded against a float64 torch reference on the same bf16 operands (rel L2), and timed synced over
`--reps` calls, arms alternating every rep, AICLK sampled from the card's sysfs node during each.
"""
import argparse, json, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack.stack import Clock          # noqa: E402

SHAPES = {"pair": (1, 288, 288, 128), "pair3": (288, 288, 128), "msa": (2, 288, 256)}


def ref_dx(x, g, gamma, eps):
    x, g, gamma = x.double(), g.double(), gamma.double()
    mean = x.mean(-1, keepdim=True)
    xc = x - mean
    rstd = (xc.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    norm = xc * rstd
    dn = g * gamma
    return (dn - dn.mean(-1, keepdim=True) - norm * (dn * norm).mean(-1, keepdim=True)) * rstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="pair,pair3,msa")
    ap.add_argument("--arms", default="composed")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "perf/bcp_device/out/lnbw_bench.json"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    clock = Clock()
    eps = 1e-5
    up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   dtype=ttnn.bfloat16, device=dev)
    arms = a.arms.split(",")
    res = {"pci": clock.pci, "shapes": {}}
    for name in a.shapes.split(","):
        shp = SHAPES[name]
        K = shp[-1]
        torch.manual_seed(0)
        x = (torch.randn(shp) * 0.7 + 0.3).bfloat16().float()
        g = (torch.randn(shp) * 1e-2).bfloat16().float()
        gamma = (1 + 0.2 * torch.randn(K)).bfloat16().float()
        ref = ref_dx(x, g, gamma, eps)
        xd, gd, gmd = up(x), up(g), up(gamma.reshape(1, K))

        def composed():
            xt = ag.Tensor(xd, requires_grad=True)
            gm = ag.Tensor(gmd, requires_grad=False)
            ag._layer_norm_bw(xt, gm, None, eps, ag.precise_config())(gd)
            return xt.grad

        def fused():
            from tt_bio import lnbw
            return lnbw.layer_norm_bw(xd, gd, gmd, eps)

        fns = {"composed": composed, "fused": fused}
        out = {}
        for arm in arms:
            dx = fns[arm]()
            ttnn.synchronize_device(dev)
            got = torch.Tensor(ttnn.to_torch(dx)).double().reshape(shp)
            out[arm] = {"rel_l2": float((got - ref).norm() / ref.norm()),
                        "max_abs": float((got - ref).abs().max()), "dtype": str(dx.dtype), "t": []}
            fns[arm]()
        for rep in range(a.reps):
            for arm in (arms if rep % 2 == 0 else arms[::-1]):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                dx = fns[arm]()
                ttnn.synchronize_device(dev)
                t1 = time.perf_counter()
                out[arm]["t"].append((t0, t1))
                del dx
        # The calls are ~ms and the clock samples every 0.25 s, so the window is the whole
        # alternated sitting, which both arms share.
        span = [(min(t[0] for v in out.values() for t in v["t"]),
                 max(t[1] for v in out.values() for t in v["t"]))]
        for arm in arms:
            ts = sorted(t1 - t0 for t0, t1 in out[arm]["t"])
            out[arm]["aiclk"] = clock.window(span)
            out[arm]["median_ms"] = 1e3 * ts[len(ts) // 2]
            out[arm]["min_ms"] = 1e3 * ts[0]
            del out[arm]["t"]
        res["shapes"][name] = out
        print(json.dumps({name: out}), flush=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    clock.stop()


if __name__ == "__main__":
    main()
