"""RFD4's triangle multiplications on a sample batch, as the trunk calls them on the float32 pair stream: device
time per design (call time / batch) including the residual write, the update's error against float64 per design,
and each design's update against the same design run alone (batch 1, first arm).

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/b128/module.py --weights <bundle> --out <json> \
        [--n 128] [--batch 16] [--arms "BASE|BASE,pair_batch"]

An arm is a lever set (comma-joined). Without `pair_batch` the trunk runs the triangle once per design
(`trunk._per_design`) and writes the residual after; with it, the whole batch is one call and the tail may take the
residual (`tri(zb, None, stream32=z)`), exactly as `PairUpdate.run` does.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tri2"))

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
from bench import reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    import ttnn
    from safetensors import safe_open

    from tt_bio import tenstorrent as T
    from tt_bio import trimul_tail as TT
    from tt_bio.rfd4.trunk import _per_design, _residual, _trimul_weights, compute_kernel_config

    f = safe_open(str(Path(a.weights) / "model.safetensors"), "pt")
    pre = "nn.pair_update_layers.0."
    state = {k[len(pre):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(pre)}
    dev = T.get_device()
    n, B = a.n, a.batch
    z = torch.nn.functional.layer_norm(torch.randn(B, n, n, 384, generator=torch.Generator().manual_seed(0),
                                                   dtype=torch.float64), (384,)).to(torch.bfloat16).float()

    def stream(zz):
        return (ttnn.from_torch(zz, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev),
                ttnn.from_torch(zz, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev))

    def call(tri, z32, zb):
        whole = int(zb.shape[0]) == 1 or T.lever("pair_batch")
        if whole and T.lever("pair_tail"):
            u = tri(zb, None, stream32=z32)
            if tri.took_residual:
                return z32, zb
        else:
            u = tri(zb, None) if whole else _per_design(tri, zb, None)
        z_new, zb_new = _residual(z32, u, ttnn.float32, None, low=ttnn.bfloat16)
        for t in (u, z32, zb):
            if t.is_allocated():
                ttnn.deallocate(t)
        return z_new, zb_new

    def update(tri, zz):
        z32, zb = stream(zz)
        z32, zb = call(tri, z32, zb)
        got = ttnn.to_torch(z32).double().reshape(zz.shape) - zz.double()
        ttnn.deallocate(z32)
        ttnn.deallocate(zb)
        return got

    rows = []
    with clocksample.during() as clk:
        for ending, name in ((False, "tri_mult_out"), (True, "tri_mult_in")):
            w64 = {k[len(name) + 1:]: v.double() for k, v in state.items() if k.startswith(name + ".")}
            ref = reference(z.double(), w64, ending)
            alone = None
            for arm in a.arms.split("|"):
                t1, t2 = list(TT.T1_STATS), list(TT.T2_STATS)
                with T.lever_scope(arm):
                    tri = T.TriangleMultiplication(ending, _trimul_weights(T.WeightScope(
                        state).child(name)), compute_kernel_config())
                    if alone is None:  # every design run alone, the batch-independence reference
                        alone = torch.cat([update(tri, z[k:k + 1]) for k in range(B)])
                    got = update(tri, z)
                    served = dict(t1=[x - y for x, y in zip(TT.T1_STATS, t1)],
                                  t2=[x - y for x, y in zip(TT.T2_STATS, t2)])
                    z32, zb = stream(z)
                    ts = []
                    for _ in range(a.reps + 3):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        z32, zb = call(tri, z32, zb)
                        ttnn.synchronize_device(dev)
                        ts.append(time.perf_counter() - t0)
                    ttnn.deallocate(z32)
                    ttnn.deallocate(zb)
                ts = sorted(ts[3:])
                err = lambda x, r: ((x - r).pow(2).flatten(1).mean(1) / r.pow(2).flatten(1).mean(1)).sqrt()  # noqa: E731
                e = err(got, ref)
                r = dict(n=n, batch=B, module=name, arm=arm, served_first_call=served,
                         ms_call=round(1e3 * ts[len(ts) // 2], 4),
                         ms_per_design=round(1e3 * ts[len(ts) // 2] / B, 4),
                         p10_ms_call=round(1e3 * ts[len(ts) // 10], 4),
                         rel_rms_vs_f64_max=float(e.max()), rel_rms_vs_f64_median=float(e.median()),
                         alone_rel_rms_vs_f64_max=float(err(alone, ref).max()),
                         max_abs_vs_alone=float((got - alone).abs().max()),
                         bit_exact_vs_alone=bool(torch.equal(got, alone)), t=time.strftime("%H:%M:%S"))
                rows.append(r)
                print(json.dumps(r), flush=True)
                del tri
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(rows=rows, aiclk=clk.summary(), script="perf/rfd4/b128/module.py"),
                                      indent=1))
    print(clk.line())


if __name__ == "__main__":
    main()
