"""RFD4's triangle multiplications as the trunk calls them, on the float32 pair stream: time per call
including the residual write, and the update's error against a float64 reference of the module.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/tri128/module.py --weights <bundle> --out <json> \
        [--n 128] [--arms "BASE|BASE,pair_tail_small"]

Each arm is a lever set (comma-joined). The trunk's no-padded-token route is used: `tri(zb, None, stream32=z)`;
where the tail took the residual the update is read back as stream - z, otherwise the trunk's `_residual` writes
it (and is timed with the call).
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
    ap.add_argument("--arms", required=True)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()

    import ttnn
    from safetensors import safe_open

    from tt_bio import tenstorrent as T
    from tt_bio import trimul_tail as TT
    from tt_bio.rfd4.trunk import _residual, _trimul_weights, compute_kernel_config

    f = safe_open(str(Path(a.weights) / "model.safetensors"), "pt")
    pre = "nn.pair_update_layers.0."
    state = {k[len(pre):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(pre)}
    dev = T.get_device()
    n = a.n
    z = torch.nn.functional.layer_norm(torch.randn(1, n, n, 384, generator=torch.Generator().manual_seed(0),
                                                   dtype=torch.float64), (384,)).to(torch.bfloat16).float()

    def stream():
        z32 = ttnn.from_torch(z, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        zb = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        return z32, zb

    def call(tri, z32, zb):
        u = tri(zb, None, stream32=z32)
        if tri.took_residual:
            return z32, zb, True
        z_new, zb_new = _residual(z32, u, ttnn.float32, None, low=ttnn.bfloat16)
        return z_new, zb_new, False

    rows = []
    with clocksample.during() as clk:
        for ending, name in ((False, "tri_mult_out"), (True, "tri_mult_in")):
            w64 = {k[len(name) + 1:]: v.double() for k, v in state.items() if k.startswith(name + ".")}
            ref = reference(z.double(), w64, ending)
            first = None
            for arm in a.arms.split("|"):
                t2 = list(TT.T2_STATS)
                with T.lever_scope(arm):
                    tri = T.TriangleMultiplication(ending, _trimul_weights(T.WeightScope(
                        state).child(name)), compute_kernel_config())
                    z32, zb = stream()
                    z32, zb, took = call(tri, z32, zb)
                    got = (ttnn.to_torch(z32).double().reshape(ref.shape) - z.double())
                    served = [x - y for x, y in zip(TT.T2_STATS, t2)]
                    ts = []
                    for _ in range(a.reps + 3):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        z32, zb, _ = call(tri, z32, zb)
                        ttnn.synchronize_device(dev)
                        ts.append(time.perf_counter() - t0)
                    ttnn.deallocate(z32)
                    ttnn.deallocate(zb)
                ts = sorted(ts[3:])
                if first is None:
                    first = got
                r = dict(n=n, module=name, arm=arm, took_residual=took, t2_served_declined=served,
                         ms=round(1e3 * ts[len(ts) // 2], 4), p10_ms=round(1e3 * ts[len(ts) // 10], 4),
                         rel_rms_vs_f64=float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt()),
                         max_abs_vs_f64=float((got - ref).abs().max()),
                         max_abs_vs_first=float((got - first).abs().max()), t=time.strftime("%H:%M:%S"))
                rows.append(r)
                print(json.dumps(r), flush=True)
                del tri
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(rows=rows, aiclk=clk.summary(), script="perf/rfd4/tri128/module.py"),
                                      indent=1))
    print(clk.line())


if __name__ == "__main__":
    main()
