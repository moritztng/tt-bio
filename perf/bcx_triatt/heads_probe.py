#!/usr/bin/env python3
"""The heads split [S,S,H*d] -> [S,H,S,d] and its inverse, as the proxy block issues it
(reshape to [S,S,H,d] + permute) and as tile-granular alternatives. Each must equal the shipped
result bit for bit, because it is a pure rearrangement."""
import argparse, json, statistics, sys, time, pathlib
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.hallgrad.e2e_distogram import ClockTrace  # noqa: E402


def timed(ttnn, dev, fn, secs, clocks):
    fn(); ttnn.synchronize_device(dev)
    t0 = time.perf_counter(); n = 0
    while time.perf_counter() - t0 < 0.3:
        fn(); n += 1
    ttnn.synchronize_device(dev)
    R = max(5, int(0.25 / ((time.perf_counter() - t0) / n)))
    ts = []; tw0 = time.time()
    while time.time() - tw0 < secs:
        t0 = time.perf_counter()
        for _ in range(R):
            fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / R)
    return statistics.median(ts) * 1e6, clocks.window(tw0, time.time())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--secs", type=float, default=2.5)
    args = ap.parse_args()
    import ttnn
    from tt_bio import tenstorrent as tt
    dev = tt.get_device()
    clocks = ClockTrace(period=0.5).start()
    S, H, d = args.n, 4, 32
    xt = torch.randn(S, S, H * d).to(torch.bfloat16)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ref_split = xt.reshape(S, S, H, d).permute(0, 2, 1, 3)
    ot = torch.randn(S, H, S, d).to(torch.bfloat16)
    o = ttnn.from_torch(ot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ref_merge = ot.permute(0, 2, 1, 3).reshape(S, S, H * d)

    split = {
        "reshape+permute": lambda: ttnn.permute(ttnn.reshape(x, [S, S, H, d]), (0, 2, 1, 3)),
        "slice4+concat": lambda: ttnn.concat(
            [ttnn.reshape(ttnn.slice(x, [0, 0, h * d], [S, S, (h + 1) * d]), [S, 1, S, d])
             for h in range(H)], dim=1),
        "nlp_create_q_kv": lambda: ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(x, [S, 1, S, H * d]), ttnn.reshape(ttnn.concat([x, x], dim=-1), [S, 1, S, 2 * H * d]),
            num_heads=H, num_kv_heads=H, transpose_k_heads=False)[0],
        "nlp_create_qkv(3x)": lambda: ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(ttnn.concat([x, x, x], dim=-1), [S, 1, S, 3 * H * d]),
            num_heads=H, num_kv_heads=H, transpose_k_heads=False)[0],
        "transpose(-2,-3) of [S,S,H,d]": lambda: ttnn.transpose(ttnn.reshape(x, [S, S, H, d]), 1, 2),
    }
    merge = {
        "permute+reshape": lambda: ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), [S, S, H * d]),
        "nlp_concat_heads": lambda: ttnn.reshape(ttnn.experimental.nlp_concat_heads(o), [S, S, H * d]),
        "concat of slices": lambda: ttnn.concat(
            [ttnn.reshape(ttnn.slice(o, [0, h, 0, 0], [S, h + 1, S, d]), [S, S, d]) for h in range(H)],
            dim=-1),
    }
    res = {"n": S, "split": {}, "merge": {}}
    for tag, fams, ref in (("split", split, ref_split), ("merge", merge, ref_merge)):
        for name, fn in fams.items():
            try:
                y = ttnn.to_torch(fn()).reshape(ref.shape)
                exact = bool(torch.equal(y, ref))
                us, clk = timed(ttnn, dev, fn, args.secs, clocks)
                res[tag][name] = dict(us=us, bitexact=exact, clock=clk)
                print(f"{tag:5s} {name:32s} {us:8.1f} us exact={exact} clk={clk}", flush=True)
            except Exception as e:
                res[tag][name] = dict(error=str(e)[:300])
                print(f"{tag:5s} {name:32s} ERROR {str(e)[:200]}", flush=True)
    clocks.stop()
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
