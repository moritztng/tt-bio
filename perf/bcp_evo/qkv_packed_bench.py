#!/usr/bin/env python3
"""triatt_bw writing dq/dk/dv into the packed qkv layout (`QKV_PACKED`) against the tape's head
merges + join, on the taped triangle attention itself.

x [B, 1, N, 3*H*32] and bias [1, H, N, N] are tape leaves, q/k/v come out of
`nlp_create_qkv_heads` on the shim, the backward is seeded with one fixed cotangent. Pass bar:
x.grad and bias.grad bit-identical arm to arm, and `packed` served on the on arm only. Then the
forward + backward timed synced with the arms alternating every rep. lead_sum_bench.py is the
template.
"""
import argparse, json, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="288x288,128x128,256x256")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "perf/bcp_evo/out/qkv_packed_bench.json"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import autograd as ag, taped_ttnn as T, triatt_bw as TB
    from tt_bio.tenstorrent import get_device
    TB.FUSED = True
    dev = get_device()
    H, d = a.heads, 32
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,  # noqa: E731
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).float()  # noqa: E731
    res = {}
    for spec in a.shapes.split(","):
        B, N = (int(v) for v in spec.split("x"))
        torch.manual_seed(0)
        x = up(torch.randn(B, 1, N, 3 * H * d))
        bias = up(torch.randn(1, H, N, N))
        seed = up(torch.randn(B, H, N, d))

        def step(packed):
            TB.QKV_PACKED = packed
            with T.tape():
                xt = ag.Tensor(x, requires_grad=True)
                bt = ag.Tensor(bias, requires_grad=True)
                q, k, v = T._SHIM.experimental.nlp_create_qkv_heads(
                    xt, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
                o = ag.triangle_attention(q, k, v, bt, scale=d ** -0.5)
            ag.backward([o], [seed])
            return xt.grad, bt.grad

        out = {}
        for arm in (False, True):
            s0 = dict(TB.STATS)
            gx, gb = step(arm)
            out[f"packed_{arm}"] = {"gx": down(gx), "gb": down(gb),
                                    "stats": {k: TB.STATS[k] - s0.get(k, 0) for k in TB.STATS},
                                    "t": []}
        off, on = out["packed_False"], out["packed_True"]
        rec = {"x_grad_equal": bool(torch.equal(off.pop("gx"), on.pop("gx"))),
               "bias_grad_equal": bool(torch.equal(off.pop("gb"), on.pop("gb")))}
        for rep in range(a.reps):
            for arm in ((False, True) if rep % 2 == 0 else (True, False)):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                r = step(arm)
                ttnn.synchronize_device(dev)
                out[f"packed_{arm}"]["t"].append(time.perf_counter() - t0)
                del r
        for arm in out.values():
            ts = sorted(arm.pop("t"))
            arm.update(median_ms=1e3 * ts[len(ts) // 2], min_ms=1e3 * ts[0])
        rec.update(out, window_utc=time.time())
        res[spec] = rec
        print(json.dumps({spec: rec}), flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
