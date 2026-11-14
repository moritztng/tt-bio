"""tenstorrent.upload (row-major upload + tilize on the chip) against the host-tilize path, by tile size.

    python perf/spd_gap_of3/upload_check.py --out out/upload [--n 512] [--reps 5]

Shapes are OpenFold3's per-fold uploads that pad 32x in TILE layout ([1, N, N, 1] pair mask / token bonds) plus a
control without padding. Reports whether each shape took the device path, bit-exactness of the device tensor read back
against `ttnn.from_torch(..., TILE_LAYOUT)`, and the median synced wall time of both paths.
"""
import argparse, json, statistics, time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    dev = T.get_device()
    n = a.n
    g = torch.Generator().manual_seed(0)
    cases = {
        "pair_mask": (torch.rand(1, n, n, 1, generator=g) > 0.1).float(),
        "token_bonds": torch.randn(1, n, n, 1, generator=g),
        "no_pad": torch.randn(1, n, n, 32, generator=g),
    }

    def timed(fn):
        ts = []
        for _ in range(a.reps):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            x = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(x)
        return statistics.median(ts)

    res = {}
    for name, t in cases.items():
        for dt_name, dt in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
            dev_path = T._device_tilize(T._tile_numel(t.shape) * T._TORCH_DT[dt].itemsize)
            host = lambda: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
            new = lambda: T.upload(t, dt, dev)
            a_h, a_n = host(), new()
            h, d = ttnn.to_torch(a_h).float(), ttnn.to_torch(a_n).float()
            ttnn.deallocate(a_h); ttnn.deallocate(a_n)
            r = {"device_path": dev_path, "shape": list(d.shape), "bit_exact": bool(torch.equal(h, d)),
                 "finite": bool(torch.isfinite(d).all()), "max_abs": float((h - d).abs().max()),
                 "host_s": timed(host), "upload_s": timed(new)}
            res[f"{name}/{dt_name}"] = r
            print(name, dt_name, r, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "result.json").write_text(json.dumps(res, indent=1))
    ok = all(r["bit_exact"] and r["finite"] for r in res.values())
    print("PASS" if ok else "FAIL", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
