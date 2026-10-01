#!/usr/bin/env python3
"""The trimul's gated channel move at the round's shapes: forward checked, backward graded and timed.

Forward: `reblock_permute_gated` against the four-way-split chain it replaces under the tape
(`ttnn.chunk`, `multiply_` with a SIGMOID b-activation, the (0,3,1,2) move), `torch.equal`.
Backward, for a cotangent `da [1, C, N, N]`: the tape entry's two arms, `composed` (permute, two
slices, sigmoid, rsub, four multiplies on stock verbs) and `fused` (`reblock_permute_gated_bw`),
each graded against float64 torch on the same bf16 operands (rel L2 and max abs, dp and dg), then
timed synced over `--reps` calls with the arms alternating every rep and AICLK sampled from the
card's sysfs node across the sitting.
"""
import argparse, json, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack.stack import Clock          # noqa: E402


def ref_bw(xw, da, p_off, g_off, C):
    xw, da = xw.double(), da.double()
    p, g = xw[..., p_off:p_off + C], xw[..., g_off:g_off + C]
    s = torch.sigmoid(g)
    dap = da.permute(0, 2, 3, 1)
    return dap * s, dap * p * s * (1 - s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="288x128,288x128f,288x64,64x128,512x128,864x128")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "perf/bcp_evo/out/gated_bw_bench.json"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import reblock_permute as R
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    clock = Clock()
    up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   dtype=ttnn.bfloat16, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).double()  # noqa: E731
    res = {"pci": clock.pci, "shapes": {}}
    for spec in a.shapes.split(","):
        # A trailing f: the cotangent arrives float32, as a promoted (two-consumer) one does.
        f32 = spec.endswith("f")
        N, C = (int(v) for v in spec.rstrip("f").split("x"))
        torch.manual_seed(0)
        # The projection's own scale at block 0 of the round, roughly: values O(1), gates O(2).
        xw = (torch.randn(1, N, N, 4 * C) * 1.5).bfloat16().float()
        da = (torch.randn(1, C, N, N) * 1e-2).bfloat16().float()
        xd = up(xw)
        dad = (ttnn.from_torch(da, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG) if f32 else up(da))
        out = {}
        # Forward: the gated kernel vs the four-way-split chain, both roles.
        q = ttnn.chunk(xd, chunks=4, dim=-1)
        eq = []
        for p_i, g_i in ((0, 1), (2, 3)):
            chain = ttnn.permute(ttnn.multiply(
                q[p_i], q[g_i], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
                (0, 3, 1, 2))
            fused = R.reblock_permute_gated(xd, p_i * C, g_i * C, C,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
            eq.append(bool(torch.equal(down(chain), down(fused))))
        out["forward_torch_equal"] = eq

        p_off, g_off = 0, C
        rdp, rdg = ref_bw(xw, da, p_off, g_off, C)

        def composed():
            dap = ttnn.permute(dad, (0, 2, 3, 1))
            pv = ttnn.slice(xd, [0, 0, 0, p_off], [1, N, N, p_off + C])
            s = ttnn.sigmoid(ttnn.slice(xd, [0, 0, 0, g_off], [1, N, N, g_off + C]))
            return (ttnn.multiply(dap, s),
                    ttnn.multiply(ttnn.multiply(dap, pv), ttnn.multiply(s, ttnn.rsub(s, 1.0))))

        def fused():
            assert R.eligible_gated_bw(dad, xd)
            return R.reblock_permute_gated_bw(dad, xd, p_off, g_off)

        fns = {"composed": composed, "fused": fused}
        for arm, fn in fns.items():
            dp, dg = fn()
            ttnn.synchronize_device(dev)
            gp, gg = down(dp).reshape(rdp.shape), down(dg).reshape(rdg.shape)
            out[arm] = {k: {"rel_l2": float((v - r).norm() / r.norm()),
                            "max_abs": float((v - r).abs().max()),
                            "ref_max": float(r.abs().max())}
                        for k, v, r in (("dp", gp, rdp), ("dg", gg, rdg))}
            out[arm]["t"] = []
        for rep in range(a.reps):
            for arm in (list(fns) if rep % 2 == 0 else list(fns)[::-1]):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                r = fns[arm]()
                ttnn.synchronize_device(dev)
                out[arm]["t"].append((t0, time.perf_counter()))
                del r
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
