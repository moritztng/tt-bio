#!/usr/bin/env python3
"""The sigmoid gate's backward at the round's shapes: graded against float64 and timed.

For ``y = o * sigmoid(g)`` and a cotangent ``d``, the two arms the tape can take: `composed` (what
`taped_ttnn._binary` runs without the lever: sigmoid of g, two multiplies, `ttnn.sigmoid_bw`) and
`fused` (`gate_bw.gate_bw`). Each is graded against float64 torch on the same operands (rel L2 and
max abs, do and dg), then timed synced over `--reps` calls with the arms alternating every rep and
AICLK sampled from the card's sysfs node across the sitting. gated_bw_bench.py is the template.
"""
import argparse, json, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack.stack import Clock          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1x288x288x128,1x288x288x128f,1x288x32x256,1x2x288x256,"
                                        "1x288x288x64,1x512x512x128,1x864x864x128,1x50x70x96")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "perf/bcp_evo/out/gate_bw_bench.json"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import gate_bw as GB
    from tt_bio.tenstorrent import get_device
    GB.GATE_BW_FUSED = True
    dev = get_device()
    clock = Clock()
    put = lambda t, dt: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=dt, device=dev,  # noqa
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).double()  # noqa: E731
    res = {"pci": clock.pci, "shapes": {}}
    for spec in a.shapes.split(","):
        f32 = spec.endswith("f")
        shape = [int(v) for v in spec.rstrip("f").split("x")]
        torch.manual_seed(0)
        o = (torch.randn(shape) * 1.5).bfloat16().float()
        g = (torch.randn(shape) * 2.0).bfloat16().float()
        d = (torch.randn(shape) * 1e-2)
        d = d if f32 else d.bfloat16().float()
        od, gd = put(o.bfloat16(), ttnn.bfloat16), put(g.bfloat16(), ttnn.bfloat16)
        dd = put(d, ttnn.float32) if f32 else put(d.bfloat16(), ttnn.bfloat16)
        s = torch.sigmoid(g.double())
        rdo, rdg = d.double() * s, d.double() * o.double() * s * (1 - s)

        def composed():
            sg = ttnn.sigmoid(gd)
            return ttnn.multiply(dd, sg), ttnn.sigmoid_bw(ttnn.multiply(dd, od), gd)[0]

        def fused():
            assert GB.eligible(dd, od, gd), GB.REJECTS
            return GB.gate_bw(dd, od, gd)

        fns = {"composed": composed, "fused": fused}
        out = {}
        for arm, fn in fns.items():
            x, y = fn()
            ttnn.synchronize_device(dev)
            out[arm] = {k: {"rel_l2": float((down(v).reshape(r.shape) - r).norm() / r.norm()),
                            "max_abs": float((down(v).reshape(r.shape) - r).abs().max()),
                            "dtype": str(v.dtype)}
                        for k, v, r in (("do", x, rdo), ("dg", y, rdg))}
            out[arm]["t"] = []
        # Inputs untouched: a writer with a wrong base address writes into whatever DRAM holds.
        out["inputs_intact"] = bool(torch.equal(down(od), o.double()) and
                                    torch.equal(down(gd), g.double()))
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
