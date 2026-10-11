"""Probe: the per-cycle MSA feature upload, host tilize vs row-major upload + device tilize.

Times each way at the shapes msa_sample uploads at c730 (depth buckets x 736 tokens x 34 channels, bf16) and
checks that the device tensors and the projection over them are bit-identical. With --load N it also runs N
busy CPU processes, the serving host's case. Prints one JSON line per (depth, way).
    TT_VISIBLE_DEVICES=<chip> python perf/spd/msa_upload_probe.py --depths 9216,2048
"""
import argparse, json, multiprocessing as mp, os, threading, time
import torch
import ttnn
from tt_bio.tenstorrent import get_device

ap = argparse.ArgumentParser()
ap.add_argument("--depths", default="9216,5120,2048")
ap.add_argument("--tokens", type=int, default=736)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--load", type=int, default=0)
a = ap.parse_args()


def burn(stop):
    x = 0
    while not stop.is_set():
        x += 1


stop = mp.Event()
burners = [mp.Process(target=burn, args=(stop,), daemon=True) for _ in range(a.load)]
for p in burners:
    p.start()

dev = get_device()
g = torch.Generator().manual_seed(0)
W = ttnn.from_torch(torch.randn(34, 128, generator=g).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                    device=dev, dtype=ttnn.bfloat16)


def host_tile(t):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)


def dev_tile(t):
    r = ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    x = ttnn.to_layout(r, ttnn.TILE_LAYOUT)
    ttnn.deallocate(r)
    return x


def gil_share(fn, t):
    """Fraction of fn's host time a second Python thread could run (1.0 = GIL released throughout)."""
    n, done = [0], threading.Event()

    def spin():
        while not done.is_set():
            n[0] += 1
    t0 = time.perf_counter(); i = 0
    while time.perf_counter() - t0 < 0.3:
        i += 1
    base_rate = i / 0.3
    th = threading.Thread(target=spin); th.start()
    t0 = time.perf_counter(); x = fn(t); h = time.perf_counter() - t0
    done.set(); th.join()
    return x, min(1.0, n[0] / (base_rate * h)), h


for D in [int(d) for d in a.depths.split(",")]:
    # One-hot over 32 tokens + has_deletion + deletion_value, as host_inputs builds it.
    tok = torch.randint(0, 32, (D, a.tokens), generator=g)
    t = torch.cat([torch.nn.functional.one_hot(tok, 32).to(torch.bfloat16),
                   (torch.rand(D, a.tokens, 1, generator=g) > 0.8).to(torch.bfloat16),
                   torch.rand(D, a.tokens, 1, generator=g).to(torch.bfloat16)], -1).unsqueeze(0).contiguous()
    outs = {}
    for name, fn in (("host_tilize", host_tile), ("device_tilize", dev_tile)):
        hs, ws, gs = [], [], []
        for r in range(a.reps + 1):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            x, share, h = gil_share(fn, t)
            ttnn.synchronize_device(dev)
            w = time.perf_counter() - t0
            if r:
                hs.append(h); ws.append(w); gs.append(share)
            if r < a.reps:
                ttnn.deallocate(x)
        y = ttnn.linear(x, W)
        outs[name] = (ttnn.to_torch(x), ttnn.to_torch(y))
        ttnn.deallocate(x); ttnn.deallocate(y)
        print(json.dumps(dict(depth=D, tokens=a.tokens, way=name, load=a.load, host_s=sorted(hs)[len(hs) // 2],
                              wall_s=sorted(ws)[len(ws) // 2], gil_free=round(sorted(gs)[len(gs) // 2], 3),
                              loadavg=os.getloadavg()[0])), flush=True)
    (xa, ya), (xb, yb) = outs["host_tilize"], outs["device_tilize"]
    print(json.dumps(dict(depth=D, x_equal=bool(torch.equal(xa.view(torch.int16), xb.view(torch.int16))),
                          proj_equal=bool(torch.equal(ya.view(torch.int16), yb.view(torch.int16))),
                          x_matches_input=bool(torch.equal(xa.view(torch.int16), t.view(torch.int16))))), flush=True)
stop.set()
