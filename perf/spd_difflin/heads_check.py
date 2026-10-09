"""Float64 error and device time of the token DiT's head re-assembly with and without the pad lanes.

    TT_VISIBLE_DEVICES=<chip> TT_BIO_LEVERS=normal python perf/spd_difflin/heads_check.py

The fp32 DiT's attention output leaves `_sdpa32` as [5, 16, 768, 64]: 16 heads of 48 channels padded to
64. Today four launches drop the pad lanes and move the heads into [5, 768, 768] (slice, permute,
reshape, permute) before the gate multiply and the output projection. `nlp_concat_heads` does it in
one launch and keeps the pads: the gate and the output projection then run at 1024 wide on weights
zero-padded in the pad lanes, exact because v's pad lanes, and so o's, are zero.
Arms: `cur` (today, dit_mm16 linears) and `cat` (concat heads, the same linears at 1024).
"""
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

M, H, S, D, DP, C = 5, 16, 768, 48, 64, 768


def err(y, r):
    e = y - r
    return f"rms {e.pow(2).mean().sqrt():.3e} max {e.abs().max():.3e}"


def timed(fn, n=20):
    dev = T.get_device()
    ttnn.deallocate(fn()); ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / n * 1e6


def main():
    dev = T.get_device()
    print("arch", T.arch_name(), flush=True)
    torch.manual_seed(11)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    o = torch.randn(M, H, S, DP)
    o[..., D:] = 0
    s = torch.randn(M, S, C).bfloat16().float()
    wg = (torch.randn(C, C) / C ** 0.5).bfloat16().float()
    wo = (torch.randn(C, C) / C ** 0.5).bfloat16().float()
    oc = o[..., :D].permute(0, 2, 1, 3).reshape(M, S, C).double()
    ref = (oc * torch.sigmoid(s.double() @ wg.double())) @ wo.double()
    relane = lambda w, ax: T._pad_head_lanes(w, H, D, DP, ax)
    f32 = lambda x: ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    b16 = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    to, ts = f32(o), b16(s)
    wgc, woc, wgp, wop = b16(wg), b16(wo), b16(relane(wg, -1)), b16(relane(wo, 0))

    def heads(cat):
        if cat:
            x = ttnn.experimental.nlp_concat_heads(to)
            return ttnn.reshape(x, (M, S, H * DP))
        x = to[:, :, :, :D]
        x = ttnn.permute(x, (0, 1, 3, 2))
        x = ttnn.reshape(x, (x.shape[0], -1, x.shape[3]))
        return ttnn.permute(x, (0, 2, 1))

    def tail(cat):
        x = heads(cat)
        g = T.k1_linear(ts, wgp if cat else wgc, None, compute_kernel_config=ckc, dtype=x.dtype)
        y = ttnn.multiply(x, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], dtype=ttnn.bfloat16)
        ttnn.deallocate(g); ttnn.deallocate(x)
        out = T.k1_linear(y, wop if cat else woc, None, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(y)
        return out

    outs = {}
    for cat in (False, True):
        name = "cat" if cat else "cur"
        try:
            outs[name] = ttnn.to_torch(tail(cat)).float()
        except Exception as e:
            print(f"{name} ERR {' | '.join(l for l in str(e).splitlines() if l.strip())[:400]}", flush=True)
            continue
        hx = ttnn.to_torch(heads(cat)).float()
        if cat:
            hx = hx.reshape(M, S, H, DP)
            print(f"cat pad lanes all zero {bool((hx[..., D:] == 0).all())}; real lanes equal cur "
                  f"{torch.equal(hx[..., :D].reshape(M, S, C), oc.float())}", flush=True)
        print(f"{name} heads {timed(lambda: heads(cat)):8.1f} us  heads+gate+out {timed(lambda: tail(cat)):8.1f} us  "
              f"vs f64 {err(outs[name].double(), ref)}", flush=True)
    if len(outs) == 2:
        print(f"cat equal cur {torch.equal(outs['cat'], outs['cur'])} wrong {(outs['cat'] != outs['cur']).sum().item()}"
              f" of {outs['cur'].numel()}", flush=True)


if __name__ == "__main__":
    main()
