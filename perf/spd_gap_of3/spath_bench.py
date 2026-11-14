#!/usr/bin/env python3
"""The OpenFold3 `--fast` DiT's six s-path linears (dit_fuse: two adaLNs' scale + shift off LN(s), the APB and
transition gates off s), 384 -> 768 at N tokens, per block vs batched across the 24 blocks of one step.

    python perf/spd_gap_of3/spath_bench.py --out OUT.json [--n 512] [--reps 20] [--blocks 24]

The p150a census (cen53, 1350 MHz) reads them at 20.3 us (bias + sigmoid epilogue) and 9.6 us (plain) a call,
0.48 s of a 10.3 s fold at 4800 block calls, against a ~4 us bandwidth roof. Arms, each one step's worth
(`--blocks` blocks), traced and replayed so the number is device time:

* `block`: today, 4 sigmoid linears with bias + 2 plain per block (`_DiTBlock._call_fused`).
* `nosig`: the same linears, bias but no epilogue (what the sigmoid epilogue costs).
* `wide`: three wide linears per step ([384, blocks*2*768] each) and one `ttnn.slice` per consumer tensor.
* `wide_noslice`: the wide linears alone (the floor if consumers could read the wide tensor in place).

* `sigapprox`: `block` with the sigmoid epilogue under math_approx_mode (the fast-mode candidate).
* `sigsep`: `nosig` + one standalone `ttnn.sigmoid` per gate (Accurate); `sigsep_fa` / `sigsep_afe` at
  SigmoidMode.FastApproximate / AccurateWithFastExp; `epi_p1` / `epi_p2` the epilogue with sigmoid params [4, 1|2].

`exact`: every sliced tensor of `wide` against `block`'s, bitwise (max_abs, n_diff).
`sig_err`: each sigmoid variant against float64 sigmoid of the float64 linear, max_abs and rms over every gate.
"""
import argparse, json, statistics, time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=24)
    a = ap.parse_args()
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio import ops
    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    g = torch.Generator().manual_seed(0)
    B, N = a.blocks, a.n
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    s = up(torch.randn(1, N, 384, generator=g))
    sh = up(torch.randn(1, N, 384, generator=g))
    # per block: (w_sc0, b_sc0, w_sb0, w_sc1, b_sc1, w_sb1, w_og, b_og, w_lg, b_lg)
    W = [[torch.randn(384, 768, generator=g) * 384 ** -0.5 for _ in range(6)] for _ in range(B)]
    Bs = [[torch.randn(1, 768, generator=g) for _ in range(4)] for _ in range(B)]
    Wd = [[up(w) for w in ws] for ws in W]
    Bd = [[up(b) for b in bs] for bs in Bs]
    ckca = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                  math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True)
    lin = lambda x, w, b=None, act=None, k=ckc: ops.linear(x, w, bias=b, activation=act,  # noqa: E731
                                                           compute_kernel_config=k, core_grid=T.CORE_GRID_MAIN)

    def block(act="sigmoid", k=ckc, post=None):
        out = []
        for w, b in zip(Wd, Bd):
            g = [lin(sh, w[0], b[0], act, k), lin(sh, w[2], b[1], act, k), lin(s, w[4], b[2], act, k),
                 lin(s, w[5], b[3], act, k)]
            if post is not None:
                g = [post(t) for t in g]
            out += [g[0], lin(sh, w[1]), g[1], lin(sh, w[3]), g[2], g[3]]
        return out

    def gates(ts):
        return [ts[i] for k in range(B) for i in (6 * k, 6 * k + 2, 6 * k + 4, 6 * k + 5)]

    # float64 reference gates from the bf16 operands the device reads
    rd = lambda t: ttnn.to_torch(t).double()  # noqa: E731
    s64, sh64 = rd(s), rd(sh)
    ref_g = []
    for k in range(B):
        w64 = [rd(Wd[k][i]) for i in (0, 2, 4, 5)]
        b64 = [rd(Bd[k][i]) for i in range(4)]
        ref_g += [torch.sigmoid(x @ w + b) for x, w, b in zip((sh64, sh64, s64, s64), w64, b64)]
    SM = ttnn.SigmoidMode
    sep = lambda m: lambda: block(None, post=lambda t: ttnn.sigmoid(t, mode=m))  # noqa: E731
    epi = lambda p: lambda: block(ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, [4.0, p]))  # noqa: E731
    variants = {"block": lambda: block(), "sigapprox": lambda: block(k=ckca),
                "sigsep": lambda: block(None, post=ttnn.sigmoid), "sigsep_fa": sep(SM.FastApproximate),
                "sigsep_afe": sep(SM.AccurateWithFastExp), "epi_p1": epi(1.0), "epi_p2": epi(2.0)}
    res_err = {}
    for name, fn in variants.items():
        try:
            gs = gates([ttnn.to_torch(t) for t in fn()])
            d = torch.cat([(x.double() - y).flatten() for x, y in zip(gs, ref_g)])
            res_err[name] = {"max_abs": float(d.abs().max()), "rms": float(d.pow(2).mean().sqrt())}
        except Exception as e:  # noqa: BLE001
            res_err[name] = {"error": str(e)[:200]}
        print("sig_err", name, res_err[name], flush=True)

    # wide: sigmoid off LN(s) = [sc0, sc1] per block, plain off LN(s) = [sb0, sb1], sigmoid off s = [og, lg]
    cat = lambda idx: up(torch.cat([W[k][i] for k in range(B) for i in idx], 1))  # noqa: E731
    catb = lambda idx: up(torch.cat([Bs[k][i] for k in range(B) for i in idx], 1))  # noqa: E731
    Wsig, bsig = cat((0, 2)), catb((0, 1))
    Wpl = cat((1, 3))
    Wg, bg = cat((4, 5)), catb((2, 3))

    def wide(slice_=True):
        ys = (lin(sh, Wsig, bsig, "sigmoid"), lin(sh, Wpl), lin(s, Wg, bg, "sigmoid"))
        if not slice_:
            return list(ys)
        out = []
        for k in range(B):
            sl = lambda y, j: ttnn.slice(y, [0, 0, (2 * k + j) * 768], [1, N, (2 * k + j + 1) * 768])  # noqa: E731
            out += [sl(ys[0], 0), sl(ys[1], 0), sl(ys[0], 1), sl(ys[1], 1), sl(ys[2], 0), sl(ys[2], 1)]
        for y in ys:
            ttnn.deallocate(y)
        return out

    res = {"n": N, "blocks": B, "reps": a.reps, "arms": {}, "sig_err": res_err}
    ref = [ttnn.to_torch(t) for t in block()]
    wd = [ttnn.to_torch(t) for t in wide()]
    d = torch.stack([(x.float() - y.float()).abs().max() for x, y in zip(ref, wd)])
    res["exact"] = {"max_abs": float(d.max()), "n_tensors_diff": int((d > 0).sum()), "n": len(ref)}
    for name, fn in (("block", block), ("nosig", lambda: block(None)), ("wide", wide),
                     ("wide_noslice", lambda: wide(False)), ("sigapprox", variants["sigapprox"]),
                     *[(k, variants[k]) for k in ("sigsep", "sigsep_fa", "sigsep_afe", "epi_p1", "epi_p2")]):
        if "error" in res_err.get(name, {}):
            continue
        for t in fn():
            ttnn.deallocate(t)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        outs = fn()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ts = []
        for _ in range(a.reps):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.release_trace(dev, tid)
        for t in outs:
            ttnn.deallocate(t)
        res["arms"][name] = {"step_ms": statistics.median(ts), "min_ms": min(ts)}
        print(name, res["arms"][name], flush=True)
    print("exact", res["exact"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
