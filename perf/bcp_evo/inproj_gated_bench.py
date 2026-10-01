#!/usr/bin/env python3
"""bcp-evo stage 14: `inproj_gated` against the path it replaces, graded and timed.

Today (`two_op`): the trimul in-projection as the model runs it (`_in_proj_matmul`, minimal_matmul
with the bias, HiFi4 float32 accumulation, the [1, N, N, 4C] result in DRAM), then the gated move
once per role (`reblock_permute_gated`). Fused: `inproj_gated` once per role. Both arms produce the
two [1, C, N, N] role outputs and are graded against float64 torch on the same bf16 operands, then
timed synced over --reps with the arms alternating every rep and AICLK sampled across the sitting.
"""
import argparse, json, os, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack.stack import Clock          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="288x128x128,300x128x128,64x128x128")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--s", default="", help="comma list of S values to sweep (TT_BIO_INPROJ_GATED_S)")
    ap.add_argument("--out", default=str(ROOT / "perf/bcp_evo/out/inproj_gated_bench.json"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import reblock_permute as R
    from tt_bio import inproj_gated as IG
    from tt_bio import tenstorrent as T
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    clock = Clock()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   dtype=ttnn.bfloat16, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).double()  # noqa: E731
    res = {"pci": clock.pci, "fidelity": str(IG.FIDELITY), "xbuf": IG.X_BUFFERS, "shapes": {}}
    s_list = [int(v) for v in a.s.split(",")] if a.s else [0]
    for spec in a.shapes.split(","):
        N, C, K = (int(v) for v in spec.split("x"))
        torch.manual_seed(0)
        x = torch.randn(1, N, N, K).bfloat16().float()          # LN'd pair, O(1)
        w = (torch.randn(K, 4 * C) * K ** -0.5).bfloat16().float()
        b = (torch.randn(4 * C) * 0.3).bfloat16().float()
        xd, wd, bd = up(x), up(w), up(b.reshape(1, -1))
        wt, ones = IG.prepare_weights(w, b, dev)
        proj = x.double() @ w.double() + b.double()
        roles = ((0, C), (2 * C, 3 * C))   # (p, g) channel offsets, a then b
        ref = [(proj[..., p:p + C] * torch.sigmoid(proj[..., g:g + C])).permute(0, 3, 1, 2)
               for p, g in roles]

        def two_op():
            gp = T._in_proj_matmul(xd, wd, ckc, ttnn.DRAM_MEMORY_CONFIG, bd)
            outs = [R.reblock_permute_gated(gp, p, g, C, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    for p, g in roles]
            ttnn.deallocate(gp)
            return outs

        def fused():
            return [IG.inproj_gated(xd, wt, ones, p, g, C) for p, g in roles]

        out = {}
        fns = {"two_op": two_op}
        for s in s_list:
            def f(s=s):
                IG.S_FORCE = s
                return fused()
            fns[f"fused_s{s}"] = f
        for arm, fn in fns.items():
            got = fn()
            ttnn.synchronize_device(dev)
            out[arm] = {}
            for name, t, r in zip("ab", got, ref):
                v = down(t).reshape(r.shape)
                out[arm][name] = {"rel_l2": float((v - r).norm() / r.norm()),
                                  "max_abs": float((v - r).abs().max()),
                                  "ref_max": float(r.abs().max()),
                                  "finite": bool(torch.isfinite(v).all())}
            out[arm]["t"] = []
            for t in got:
                ttnn.deallocate(t)
        for rep in range(a.reps):
            for arm in (list(fns) if rep % 2 == 0 else list(fns)[::-1]):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                r = fns[arm]()
                ttnn.synchronize_device(dev)
                out[arm]["t"].append((t0, time.perf_counter()))
                for t in r:
                    ttnn.deallocate(t)
        span = [(min(t[0] for k in fns for t in out[k]["t"]),
                 max(t[1] for k in fns for t in out[k]["t"]))]
        for arm in fns:
            ts = sorted(t1 - t0 for t0, t1 in out[arm].pop("t"))
            out[arm].update(aiclk=clock.window(span), median_ms=1e3 * ts[len(ts) // 2],
                            min_ms=1e3 * ts[0])
        res["shapes"][spec] = out
        print(json.dumps({spec: out}), flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    clock.stop()


if __name__ == "__main__":
    main()
