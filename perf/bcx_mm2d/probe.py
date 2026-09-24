#!/usr/bin/env python3
"""bcx-mm2d probe: rank-3 matmul vs its 2D view, at every linear shape the census block issues.

For each shape: time the shipped rank-3 call, the same call on a reshaped 2D view (reshape in, call,
reshape out, all timed), and each reshape alone. Then compare the two outputs byte for byte and
grade both against a float64 product of the same bf16 operands. HiFi4 + fp32 acc, the config
autograd uses. 3 warm-up calls, 20 timed, one sync, 5 repeats.
"""
import json, pathlib, statistics, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from perf.hallgrad.e2e_distogram import ClockTrace  # noqa: E402
from perf.hallgrad.census import stamp  # noqa: E402


def timed(ttnn, dev, fn):
    ts = []
    for _ in range(5):
        for _ in range(3):
            fn()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / 20 * 1e6)
    return {"median_us": statistics.median(ts), "min_us": min(ts), "max_us": max(ts)}


def main():
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    dev = tt.get_device()
    clocks = ClockTrace(period=1.0).start()
    cfg = ag.precise_config()
    torch.manual_seed(0)

    def T(*s):
        h = torch.randn(*s).to(torch.bfloat16)
        return h, ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    # (name, x shape, w shape, transpose_b, op) -- the census's linear-family rows
    shapes = [
        ("fwd 128->128  (census row 1)", (256, 256, 128), (128, 128), False, "linear"),
        ("dX  128->128  (census row 2)", (256, 256, 128), (128, 128), True, "matmul"),
        ("dX  512->128  (census row 11)", (256, 256, 512), (128, 512), True, "matmul"),
        ("fwd 512->128  (census row 22)", (256, 256, 512), (512, 128), False, "linear"),
        ("fwd 128->512  (census row 24)", (256, 256, 128), (128, 512), False, "linear"),
        ("dX  4->128    (census row 31)", (256, 256, 4), (128, 4), True, "matmul"),
        ("fwd 128->4", (256, 256, 128), (128, 4), False, "linear"),
        ("fwd 128->128 rank-4 [1,N,N,c]", (1, 256, 256, 128), (128, 128), False, "linear"),
        ("fwd 128->128 via ops.linear", (256, 256, 128), (128, 128), False, "ops.linear"),
    ]
    out = {"stamp": stamp(), "cases": {}}
    t_start = time.time()
    for name, xs, ws, tb, op in shapes:
        xh, x3 = T(*xs)
        wh, w = T(*ws)
        lead = 1
        for d in xs[:-1]:
            lead *= d
        if op == "ops.linear":
            from tt_bio import ops
            call = lambda a: ops.linear(a, w, compute_kernel_config=cfg)
        elif op == "linear":
            call = lambda a: ttnn.linear(a, w, compute_kernel_config=cfg)
        else:
            call = lambda a: ttnn.matmul(a, w, transpose_b=tb, compute_kernel_config=cfg)
        x2 = ttnn.reshape(x3, [lead, xs[-1]])
        same_buf = x2.buffer_address() == x3.buffer_address()
        rec = {"x": list(xs), "w": list(ws), "transpose_b": tb, "op": op,
               "flatten_is_view": same_buf}
        rec["rank3"] = timed(ttnn, dev, lambda: call(x3))
        rec["view2d_call_only"] = timed(ttnn, dev, lambda: call(x2))
        n_out = ws[0] if tb else ws[1]
        rec["view2d_with_reshapes"] = timed(
            ttnn, dev, lambda: ttnn.reshape(call(ttnn.reshape(x3, [lead, xs[-1]])), [*xs[:-1], n_out]))
        rec["reshape_in_only"] = timed(ttnn, dev, lambda: ttnn.reshape(x3, [lead, xs[-1]]))
        y3 = call(x3)
        y2 = ttnn.reshape(call(x2), [*xs[:-1], n_out])
        a, b = ttnn.to_torch(y3), ttnn.to_torch(y2)
        rec["bit_identical"] = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
        rec["n_differing"] = int((a.view(torch.int16) != b.view(torch.int16)).sum())
        ref = xh.double() @ (wh.double().T if tb else wh.double())
        rb = ref.to(torch.bfloat16)   # the correctly rounded bf16 answer

        def ordered(t):   # bf16 bits -> integers that count ulps across zero
            i = t.contiguous().view(torch.int16).int()
            return torch.where(i < 0, -(i & 0x7FFF), i)
        rec["max_ulp_between_arms"] = int((ordered(a) - ordered(b)).abs().max())
        for k, v in (("rank3", a), ("view2d", b)):
            rec[f"frac_correctly_rounded_{k}"] = float((v.view(torch.int16) == rb.view(torch.int16)).double().mean())
            rec[f"max_ulp_vs_rounded_{k}"] = int((ordered(v) - ordered(rb)).abs().max())
        # Same two programs with an fp32 result: if these agree and the bf16 ones do not, the
        # arms differ only in where the running sum is rounded to bf16, not in what they sum.
        if op != "ops.linear":
            kw32 = dict(compute_kernel_config=cfg, dtype=ttnn.float32)
            c32 = (lambda t: ttnn.linear(t, w, **kw32)) if op == "linear" else \
                  (lambda t: ttnn.matmul(t, w, transpose_b=tb, **kw32))
            a32 = ttnn.to_torch(c32(x3)).double()
            b32 = ttnn.to_torch(ttnn.reshape(c32(x2), [*xs[:-1], n_out])).double()
            rec["fp32_out_bit_identical"] = bool(torch.equal(a32, b32))
            rec["fp32_out_rel_l2_between_arms"] = float((a32 - b32).norm() / a32.norm())
            rec["fp32_out_rel_l2_vs_f64_rank3"] = float((a32 - ref).norm() / ref.norm())
            rec["fp32_out_rel_l2_vs_f64_view2d"] = float((b32 - ref).norm() / ref.norm())
            d = v.double() - ref
            rec[f"rel_l2_vs_f64_{k}"] = float(d.norm() / ref.norm())
            rec[f"max_abs_vs_f64_{k}"] = float(d.abs().max())
        out["cases"][name] = rec
        print(f"{name:32s} rank3 {rec['rank3']['median_us']:7.1f}  2D {rec['view2d_call_only']['median_us']:7.1f}  "
              f"2D+reshapes {rec['view2d_with_reshapes']['median_us']:7.1f}  reshape {rec['reshape_in_only']['median_us']:5.1f} us  "
              f"view={same_buf} bitexact={rec['bit_identical']} ({rec['n_differing']} diff)  "
              f"relL2 {rec['rel_l2_vs_f64_rank3']:.3e}/{rec['rel_l2_vs_f64_view2d']:.3e} "
              f"rounded {rec['frac_correctly_rounded_rank3']:.4f}/{rec['frac_correctly_rounded_view2d']:.4f} "
              f"ulp {rec['max_ulp_vs_rounded_rank3']}/{rec['max_ulp_vs_rounded_view2d']} arms {rec['max_ulp_between_arms']} "
              f"fp32-out identical {rec.get('fp32_out_bit_identical')} relL2 {rec.get('fp32_out_rel_l2_vs_f64_rank3', 0):.2e}/{rec.get('fp32_out_rel_l2_vs_f64_view2d', 0):.2e}", flush=True)
    # the costly reshape, for contrast: a reshape that changes the last two (tiled) dims
    _, h = T(256, 256, 128)
    out["heads_reshape_256x256x128_to_x4x32"] = timed(ttnn, dev, lambda: ttnn.reshape(h, [256, 256, 4, 32]))
    print("heads reshape", out["heads_reshape_256x256x128_to_x4x32"], flush=True)
    clocks.stop()
    out["clock_window"] = clocks.window(t_start, time.time() + 1)
    out["clock_meta"] = clocks.summary()
    print("AICLK", out["clock_window"])
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
