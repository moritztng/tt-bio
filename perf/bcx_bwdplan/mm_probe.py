#!/usr/bin/env python3
"""The backward matmuls `bcx-realcensus` found on 1 to 8 cores, one plan per variant.

Each case is a VJP product exactly as `taped_ttnn._v_matmul` / `autograd.linear` issue it at
n=256, on random operands of the real shapes and dtypes. Variants:

  shipped   ttnn's own plan (no program config), what runs today
  bmm       `bmm_program_config` over PADDED tile counts (logical dims rounded up to 32)
  xpose     the transposed operand made explicit with `ttnn.transpose`, then the plain product
            under `bmm` (the forward's plan shape)
  flat      a 2-D right operand: the left operand's leading dims folded into M by a reshape

Graded against float64 on the same bf16 operands; timed in windows long enough for an AICLK
sample from the card's own sysfs node.
"""
import argparse, json, statistics, sys, time, pathlib
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack.stack import Clock  # noqa: E402


def padded_cfg(ttnn, a, b, ta, tb, out_tiles=64):
    sa, sb = [int(d) for d in a.shape], [int(d) for d in b.shape]
    if len(sa) < 3 or len(sa) != len(sb) or sa[:-2] != sb[:-2]:
        return None
    t = lambda x: -(-x // 32)
    M, K = (sa[-1], sa[-2]) if ta else (sa[-2], sa[-1])
    N = sb[-2] if tb else sb[-1]
    Mt, Nt, Kt = t(M), t(N), t(K)
    if Mt * Nt > out_tiles:
        return None
    ld = lambda n, cap: max(d for d in range(1, min(n, cap) + 1) if n % d == 0)
    sw = ld(Nt, 4)
    sh = ld(Mt, max(1, 4 // sw))
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=a.device().compute_with_storage_grid_size(),
        in0_block_w=ld(Kt, 8), out_subblock_h=sh, out_subblock_w=sw, per_core_M=Mt, per_core_N=Nt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--secs", type=float, default=1.5)
    ap.add_argument("--only", default=None, help="case-name prefix")
    args = ap.parse_args()
    import ttnn
    from tt_bio import tenstorrent as tt, af2
    dev = tt.get_device()
    clock = Clock()
    cfg = af2.compute_kernel_config()
    g = torch.Generator().manual_seed(0)
    host = {}

    def T(name, *s, scale=1.0):
        x = (torch.randn(*s, generator=g) * scale).to(torch.bfloat16)
        host[name] = x
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    cases = {}
    # MSA column attention at depth 1: q [L,H,1,d], k^T [L,H,d,1], scores/probs [L,H,1,1]
    L, H, d = 256, 8, 32
    q, kt = T("c.q", L, H, 1, d), T("c.kt", L, H, d, 1)
    gs, p = T("c.gs", L, H, 1, 1), T("c.p", L, H, 1, 1)
    v, go = T("c.v", L, H, 1, d), T("c.go", L, H, 1, d)
    cases["col dq=gs@kt^T"] = (gs, kt, False, True)
    cases["col dkt=q^T@gs"] = (q, gs, True, False)
    cases["col dp=go@v^T"] = (go, v, False, True)
    cases["col dv=p^T@go"] = (p, go, True, False)
    # triangle attention, one 81-row block of `_fp32_softmax_attention` at n=256, 4 heads
    for B in (81, 13):
        q, kt = T(f"t{B}.q", B, 4, 256, 32), T(f"t{B}.kt", B, 4, 32, 256)
        gs, p = T(f"t{B}.gs", B, 4, 256, 256, scale=1 / 16), T(f"t{B}.p", B, 4, 256, 256, scale=1 / 16)
        v, go = T(f"t{B}.v", B, 4, 256, 32), T(f"t{B}.go", B, 4, 256, 32)
        cases[f"tri{B} dq=gs@kt^T"] = (gs, kt, False, True)
        cases[f"tri{B} dkt=q^T@gs"] = (q, gs, True, False)
        cases[f"tri{B} dp=go@v^T"] = (go, v, False, True)
        cases[f"tri{B} dv=p^T@go"] = (p, go, True, False)
    # MSA-track row linears at depth 1 (column attention): dx = g @ W^T
    for n_out in (768, 256):
        gl, w = T(f"l{n_out}.g", 256, 1, n_out), T(f"l{n_out}.w", 256, n_out, scale=1 / 16)
        cases[f"lin dx=g@W^T ({n_out})"] = (gl, w, False, True)
    # the same linear at a tile-row shape, where `_via2d` already collapses it
    gl, w = T("lt.g", 256, 256, 768), T("lt.w", 256, 768, scale=1 / 16)
    cases["lin dx=g@W^T tile rows"] = (gl, w, False, True)
    if args.only:
        cases = {k: v for k, v in cases.items() if k.startswith(args.only)}

    res = {"stamp": {"t": time.time(), "pci": clock.pci}, "cases": {}}
    for name, (a, b, ta, tb) in cases.items():
        res["cases"][name] = run_case(ttnn, dev, clock, cfg, a, b, ta, tb, args.secs)
    clock.stop()
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


def run_case(ttnn, dev, clock, cfg, a, b, ta, tb, secs):
    A64 = ttnn.to_torch(a).double()
    B64 = ttnn.to_torch(b).double()
    ref = (A64.transpose(-1, -2) if ta else A64) @ (B64.transpose(-1, -2) if tb else B64)
    variants = {"shipped": lambda: ttnn.matmul(a, b, transpose_a=ta, transpose_b=tb,
                                               compute_kernel_config=cfg)}
    pc = padded_cfg(ttnn, a, b, ta, tb)
    if pc is not None:
        variants["bmm"] = lambda: ttnn.matmul(a, b, transpose_a=ta, transpose_b=tb,
                                              compute_kernel_config=cfg, program_config=pc)
    if len(a.shape) == len(b.shape):
        def xpose():
            aa = ttnn.transpose(a, -2, -1) if ta else a
            bb = ttnn.transpose(b, -2, -1) if tb else b
            c = padded_cfg(ttnn, aa, bb, False, False)
            return ttnn.matmul(aa, bb, compute_kernel_config=cfg, program_config=c)
        variants["xpose"] = xpose
    if len(b.shape) == 2 and not ta:
        s = [int(x) for x in a.shape]

        def flat():
            y = ttnn.matmul(ttnn.reshape(a, [s[0] * s[1], s[2]]) if len(s) == 3 else a, b,
                            transpose_b=tb, compute_kernel_config=cfg)
            return ttnn.reshape(y, s[:-1] + [int(y.shape[-1])])
        variants["flat"] = flat
        wt = ttnn.transpose(b, -2, -1) if tb else b
        variants["wT"] = lambda: ttnn.matmul(a, ttnn.transpose(b, -2, -1) if tb else b,
                                             compute_kernel_config=cfg)
        variants["wT_cached"] = lambda: ttnn.matmul(a, wt, compute_kernel_config=cfg)
    out, base = {}, None
    for vn, fn in variants.items():
        try:
            y = ttnn.to_torch(fn()).double().reshape(ref.shape)
            rel = float((y - ref).norm() / ref.norm())
            base = y if base is None else base
            exact = bool(torch.equal(y, base))
            fn(); ttnn.synchronize_device(dev)
            t0 = time.perf_counter(); n = 0
            while time.perf_counter() - t0 < 0.2:
                fn(); n += 1
            ttnn.synchronize_device(dev)
            R = max(3, int(0.25 / ((time.perf_counter() - t0) / n)))
            ts, tw0 = [], time.time()
            while time.time() - tw0 < secs:
                t0 = time.perf_counter()
                for _ in range(R):
                    fn()
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) / R)
            tw1 = time.time()
            us = statistics.median(ts) * 1e6
            out[vn] = dict(us=us, n=len(ts), R=R, rel_l2_vs_f64=rel, bitexact_vs_shipped=exact,
                           aiclk=clock.window([(tw0, tw1)]))
            print(f"{str(list(a.shape))+'x'+str(list(b.shape)):40s} ta={ta:d} tb={tb:d} {vn:8s} "
                  f"{us:9.1f} us relL2={rel:.3e} exact={exact} clk={out[vn]['aiclk']}", flush=True)
        except Exception as e:
            out[vn] = dict(error=str(e)[:400])
            print(f"{vn} ERROR {str(e)[:300]}", flush=True)
    return out


if __name__ == "__main__":
    main()
