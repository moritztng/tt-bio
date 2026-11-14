"""Per-site precision bench for RFD4's pair-track matmuls: device time per call and error vs float64.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/bfp8/sites.py --weights <bundle> --out <json> [--n 128 512]

Sites are the five matmuls a pair update runs per pair element, with layer 0's checkpoint weights:
trimul in-projection (a_p, b_p, a_g, b_g packed, 384 -> 1024), trimul g_out (384 -> 384), trimul
p_out (256 -> 384), pair transition fc1 (384 -> 768, relu) and fc2 (768 -> 384). Each arm is a
(weight dtype, activation dtype, output dtype, math fidelity) on the trunk's fp32-dest config.
Error is relative RMS against the float64 product of the same bf16 input and the fp32 weights,
so the bf16 HiFi4 base carries the floor every arm is read against. Time is the median of
`--reps` synced calls after warm-up. AICLK is sampled during the run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402

ROWS = 16384
ARMS = [  # name, weight, activation, output, fidelity
    ("bf16_hifi4", "bf16", "bf16", "bf16", "HiFi4"),
    ("bf16_hifi2", "bf16", "bf16", "bf16", "HiFi2"),
    ("bf16_lofi", "bf16", "bf16", "bf16", "LoFi"),
    ("w8_hifi2", "b8", "bf16", "bf16", "HiFi2"),
    ("w8_lofi", "b8", "bf16", "bf16", "LoFi"),
    ("w8_hifi2_o8", "b8", "bf16", "b8", "HiFi2"),
    ("w8x8_hifi2", "b8", "b8", "bf16", "HiFi2"),
    ("w8x8_lofi", "b8", "b8", "bf16", "LoFi"),
]


def sites(weights):
    from safetensors import safe_open

    f = safe_open(str(Path(weights) / "model.safetensors"), "pt")
    g = lambda k: f.get_tensor("nn.pair_update_layers.0." + k).double()  # noqa: E731
    tm = "tri_mult_out."
    return [
        ("tm_in", torch.cat([g(tm + s) for s in ("linear_a_p.weight", "linear_b_p.weight",
                                                    "linear_a_g.weight", "linear_b_g.weight")]), None, None),
        ("tm_gout", g(tm + "linear_g.weight"), None, None),
        ("tm_pout", g(tm + "linear_z.weight"), None, None),
        ("tr_fc1", g("transition_out.linear_1.weight"), g("transition_out.linear_1.bias"), "relu"),
        ("tr_fc2", g("transition_out.linear_2.weight"), g("transition_out.linear_2.bias"), None),
    ]


def inputs(name, n, k, w_fc1, b_fc1, gen):
    """A bf16 input shaped like the site's: layer-normed pair rows, or relu(fc1) for fc2."""
    x = torch.randn(n * n, 384 if name == "tr_fc2" else k, generator=gen, dtype=torch.float64)
    x = torch.nn.functional.layer_norm(x, x.shape[-1:])
    if name == "tr_fc2":
        x = torch.relu(x @ w_fc1.t() + b_fc1)
    return x.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    import ttnn

    from tt_bio.rfd4.trunk import compute_kernel_config
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    base = compute_kernel_config()
    dt = {"bf16": ttnn.bfloat16, "b8": ttnn.bfloat8_b}

    def ckc(fid):
        c = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=base.math_approx_mode,
            fp32_dest_acc_en=base.fp32_dest_acc_en, packer_l1_acc=base.packer_l1_acc)
        return c

    up = lambda t, d: ttnn.from_torch(t.float().contiguous(), dtype=dt[d], layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                      device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    S = sites(a.weights)
    w_fc1, b_fc1 = S[3][1], S[3][2]
    rows = []
    with clocksample.during() as clk:
        for n in a.n:
            for name, w, b, act in S:
                gen = torch.Generator().manual_seed(0)
                k = int(w.shape[1])
                x = inputs(name, n, k, w_fc1, b_fc1, gen)
                xs = x[:ROWS].double()   # a linear is row-independent: grade a row sample, time all rows
                ref = xs @ w.t() + (0 if b is None else b)
                if act == "relu":
                    ref = torch.relu(ref)
                for arm, wd, xd, od, fid in ARMS:
                    xt, wt = up(x, xd), up(w.t(), wd)
                    bt = None if b is None else up(b[None], "bf16")
                    kw = dict(bias=bt, compute_kernel_config=ckc(fid), dtype=dt[od], activation=act,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    try:
                        y = ttnn.linear(xt, wt, **kw)
                    except RuntimeError as e:   # the default program refuses the shape (L1)
                        r = dict(n=n, site=name, arm=arm, refused=str(e).splitlines()[0][:200])
                        rows.append(r)
                        print(json.dumps(r), flush=True)
                        for t in (xt, wt, bt):
                            if t is not None:
                                ttnn.deallocate(t)
                        continue
                    got = ttnn.to_torch(y).reshape(-1, ref.shape[-1])[:ROWS].double()
                    ttnn.deallocate(y)
                    err = float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
                    ts = []
                    for _ in range(a.reps + 3):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        y = ttnn.linear(xt, wt, **kw)
                        ttnn.synchronize_device(dev)
                        ts.append(time.perf_counter() - t0)
                        ttnn.deallocate(y)
                    ts = sorted(ts[3:])
                    for t in (xt, wt, bt):
                        if t is not None:
                            ttnn.deallocate(t)
                    r = dict(n=n, site=name, arm=arm, k=k, m=int(w.shape[0]), ms=1e3 * ts[len(ts) // 2],
                             ms_min=1e3 * ts[0], rel_rms_vs_f64=err, t=time.strftime("%H:%M:%S"))
                    rows.append(r)
                    print(json.dumps(r), flush=True)
    out = dict(rows=rows, aiclk=clk.summary(), load=clk.load and max(clk.load), script="perf/rfd4/bfp8/sites.py")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("aiclk", clk.summary())


if __name__ == "__main__":
    main()
