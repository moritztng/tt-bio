"""Price the two trace additions the inference A/B found, on the card the matrix ran on.

protenix-v2: AFTER dispatches 484 more `ttnn.clamp(d, -60.0, None)` per fold, one inside every
`_accurate_softmax` call, all on (1, 16, 128, 128) fp32 (commit f84e232af, the -60 floor).
openfold3: AFTER uploads 16 confidence-head weights in `__init__` that inference never applies
(commit 0802a53f0, `_DEVICE_HEAD_WEIGHTS`).

Each is timed as a synchronized loop of exactly the per-fold count, so the number is the wall a
fold pays for it, dispatch included, which bounds its device time from above. The clamp is set
beside the `_accurate_softmax` chain and the fused `ttnn.softmax` at the same shape. AICLK is
sampled from tt-smi during the timing. Run from the AFTER tree (cwd and PYTHONPATH).
"""
import json, statistics, subprocess, sys, threading, time
import torch, ttnn
from tt_bio import tenstorrent as T

SMI = "/home/ttuser/.local/bin/tt-smi"
CARD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
REPS = 7
# (shape, transpose) of the 16 uploads, in the order the AFTER trace lists them.
HEAD_UPLOADS = [((128, 449), True), ((128, 449), True), ((128, 39), True), ((64, 128), True),
                ((128,), False), ((128,), False), ((64, 128), True), ((128,), False),
                ((128,), False), ((64, 128), True), ((384,), False), ((384,), False),
                ((1150, 384), True), ((384,), False), ((384,), False), ((46, 384), True)]


def clock(stop, out):
    while not stop.is_set():
        try:
            d = json.loads(subprocess.run([SMI, "-s"], capture_output=True, text=True,
                                          timeout=20).stdout)
            out.append(int(d["device_info"][CARD]["telemetry"]["aiclk"].strip()))
        except Exception:
            pass
        stop.wait(1)


def timed(fn, dev):
    fn()  # warm: compiles, fills the program cache
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(ts), "min_s": min(ts), "all_s": ts}


def main(out):
    dev = T.get_device()
    g = torch.Generator().manual_seed(0)
    x = ttnn.from_torch(torch.randn(1, 16, 128, 128, generator=g) * 8, layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.float32)
    xb = ttnn.typecast(x, ttnn.bfloat16)

    def clamps():
        for _ in range(484):
            y = ttnn.clamp(x, -60.0, None)
            ttnn.deallocate(y)

    def chains():
        for _ in range(484):
            ttnn.deallocate(T._accurate_softmax(xb))

    def fused():
        for _ in range(484):
            ttnn.deallocate(ttnn.softmax(xb, dim=-1))

    ws = [(torch.randn(*s, generator=g), t) for s, t in HEAD_UPLOADS]

    def uploads():
        for w, t in ws:
            v = ttnn.from_torch(w.t().contiguous() if t else w, layout=ttnn.TILE_LAYOUT,
                                device=dev, dtype=ttnn.bfloat16)
            ttnn.deallocate(v)

    stop, clk = threading.Event(), []
    th = threading.Thread(target=clock, args=(stop, clk), daemon=True)
    th.start()
    res = {"card": CARD, "tree": T.__file__, "reps": REPS,
           "protenix_484_clamp": timed(clamps, dev),
           "protenix_484_accurate_softmax_chain": timed(chains, dev),
           "protenix_484_fused_softmax": timed(fused, dev),
           "openfold3_16_head_uploads": timed(uploads, dev)}
    stop.set()
    th.join()
    s = sorted(clk)
    res["aiclk_mhz_sampled_DURING"] = ({"n": len(s), "min": s[0], "median": s[len(s) // 2],
                                        "max": s[-1]} if s else None)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: (v["median_s"] if isinstance(v, dict) and "median_s" in v else v)
                      for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
