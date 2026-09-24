#!/usr/bin/env python3
"""The backwards of the taped `nlp_concat_heads` and `nlp_create_qkv_heads`, old expression
against new, bit for bit and timed, at the n=256 pair shape (B=L=256, 4 heads x 32), in bf16
and fp32 gradients. Both are pure rearrangements, so anything short of bit-identical is a bug."""
import argparse, json, statistics, sys, time, pathlib
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.hallgrad.e2e_distogram import ClockTrace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    assert ag.__file__.startswith(str(ROOT))
    dev = tt.get_device()
    clocks = ClockTrace(period=0.5).start()
    B = L = args.n
    H, dh = 4, 32
    res = {}

    def timeit(fn, secs=2.0):
        fn(); ttnn.synchronize_device(dev)
        ts = []; t_w = time.time()
        while time.time() - t_w < secs:
            t0 = time.perf_counter()
            for _ in range(5):
                fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) / 5)
        return statistics.median(ts) * 1e6, clocks.window(t_w, time.time())

    for tdt, dt in ((torch.bfloat16, ttnn.bfloat16), (torch.float32, ttnn.float32)):
        gc = ttnn.from_torch(torch.randn(B, 1, L, H * dh).to(tdt), dtype=dt,
                             layout=ttnn.TILE_LAYOUT, device=dev)
        gq = ttnn.from_torch(torch.randn(B, H, L, dh).to(tdt), dtype=dt,
                             layout=ttnn.TILE_LAYOUT, device=dev)
        s = 1
        zero_old = ttnn.zeros([B, L, 1, H * dh], dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        zero_new = ttnn.zeros([B, 1, L, H * dh], dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        arms = {
            "concat_heads_bw": (
                lambda: ttnn.permute(ttnn.reshape(gc, [B, L, H, dh]), [0, 2, 1, 3]),
                lambda: ag._split_heads_v(ttnn.reshape(gc, [B, L, H * dh]), H)),
            "create_qkv_heads_bw": (
                lambda: ttnn.reshape(ttnn.concat(
                    [ttnn.reshape(ttnn.permute(gq, [0, 2, 1, 3]), [B, L, 1, H * dh])
                     if i == s else zero_old for i in range(3)], dim=2), [B, 1, L, 3 * H * dh]),
                lambda: ttnn.concat(
                    [ttnn.reshape(ag._merge_heads_v(gq), [B, 1, L, H * dh])
                     if i == s else zero_new for i in range(3)], dim=-1)),
        }
        for name, (old, new) in arms.items():
            a, b = ttnn.to_torch(old()), ttnn.to_torch(new())
            exact = bool(torch.equal(a.reshape(b.shape), b))
            t_old, c_old = timeit(old)
            t_new, c_new = timeit(new)
            key = f"{name}:{str(dt).split('.')[-1]}"
            res[key] = dict(bitexact=exact, old_us=t_old, new_us=t_new, clock_old=c_old,
                            clock_new=c_new)
            print(f"{key:32s} exact={exact} old {t_old:8.1f} us  new {t_new:8.1f} us  "
                  f"{t_old / t_new:5.2f}x  clk {c_old.get('median')}/{c_new.get('median')}", flush=True)
    clocks.stop()
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
