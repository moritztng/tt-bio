"""Mol-1 pair transition (and width adapter) at its two trunk widths: ms/call and error vs float64, per lever set.

Usage: TT_VISIBLE_DEVICES=<card> python perf/mol1_pairmix/transition_bench.py CKPT N OUT_JSON [SET=lever,lever ...]
Block 0 (PairMixer, c_z 448) and block 8 (c_z 320) pair transitions with Mol-1's own weights, called the way
PairformerLayer calls them (residual added in place); input is randn * 4. Each SET name is timed over 20 warm
calls; `base` is protenix.NORMAL_LEVERS. Prints the AICLK sampled during the timed calls.
"""
import json
import sys
import time

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import ttnn  # noqa: E402

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", ".."))
from perf.clocksample import during  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
import tt_bio.protenix as P  # noqa: E402
from tt_bio import protenix_weights as PW  # noqa: E402


def ref(x, w):
    x = x.double()
    xn = F.layer_norm(x, x.shape[-1:], w["norm.weight"].double(), w["norm.bias"].double(), 1e-5)
    h = F.silu(F.linear(xn, w["fc1.weight"].double())) * F.linear(xn, w["fc2.weight"].double())
    return x + F.linear(h, w["fc3.weight"].double())


def main(ckpt, n, out, *sets):
    n = int(n)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    arms = {"base": P.NORMAL_LEVERS}
    for s in sets:
        name, _, extra = s.partition("=")
        arms[name] = P.NORMAL_LEVERS | frozenset(e for e in extra.split(",") if e)
    res = {"n": n, "arms": {}}
    for blk in (0, 8):
        p = f"pairformer_stack.blocks.{blk}."
        w = {k[len("transition_z."):]: v.float() for k, v in
             PW.remap_pairformer_block({k[len(p):]: v for k, v in sd.items() if k.startswith(p)}).items()
             if k.startswith("transition_z.")}
        c = w["norm.weight"].shape[0]
        g = torch.Generator().manual_seed(blk)
        x = torch.randn(1, n, n, c, generator=g) * 4
        y64 = ref(x, w)
        for name, lv in arms.items():
            with T.levers(lv):
                mod = T.Transition(w, ckc)
                xt = T.upload(x, ttnn.bfloat16, dev)
                yt = mod(xt, add_to_input=True)
                y = torch.Tensor(T.download(yt)).double().reshape(y64.shape)
                ttnn.deallocate(yt)
                err = float((y - y64).norm() / (y64 - x.double()).norm())
                xs = [T.upload(x, ttnn.bfloat16, dev) for _ in range(3)]
                ttnn.synchronize_device(dev)
                times = []
                with during() as clk:
                    for i in range(20):
                        xi = ttnn.clone(xs[i % 3])
                        ttnn.synchronize_device(dev)
                        t = time.perf_counter()
                        yi = mod(xi, add_to_input=True)
                        ttnn.synchronize_device(dev)
                        times.append((time.perf_counter() - t) * 1e3)
                        ttnn.deallocate(yi)
                for xx in xs:
                    ttnn.deallocate(xx)
            times.sort()
            row = {"c": c, "ms_median": round(times[len(times) // 2], 3), "ms_min": round(times[0], 3),
                   "rel_err_update_vs_f64": round(err, 5), "clock": clk.line()}
            res["arms"][f"{c}:{name}"] = row
            print(c, name, row, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
