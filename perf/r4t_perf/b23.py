"""r4t-perf L1, B2/B3: the elementwise optimizer update on device (fp32) vs the host step in wk/r4t-scale optim.py.

B2 Muon (given the same orthogonalized O): buf <- buf + (1-mu)(g - buf); u = buf - (buf - g)(1-mu) (nesterov);
   theta <- theta * (1 - lr wd) + (-lr ratio) O. torch's lerp op order, float32 constants, as AdamW._muon_new.
B3 AdamW leaves: m <- b1 m + (1-b1) g; v <- b2 v + (1-b2) g^2; theta -= lr ((m/bc1)/(sqrt(v/bc2)+eps) + wd theta).
Bar (pre-registered rows/perf.md, interpretation fixed 06:25Z before this ran): per tensor max|dev-host|/max|host|
<= 1e-6 for theta, buf, m, v; bit-identical fraction reported. --broken skips one Muon tensor's update: must FAIL.
--time: whole-model sizes (RFD4 denoiser 297 Muon matrices + 179,456 Adam elements), synced, AICLK during.
"""
import argparse, json, os, subprocess, threading, time
import numpy as np

MU, LR, WD_M, B1, B2, EPS, WD_A = 0.95, 2e-4, 0.1, 0.9, 0.999, 1e-8, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--broken", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch, ttnn
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
    sha = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    f = np.float32
    put = lambda x: ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x)), dtype=ttnn.float32,
                                    layout=ttnn.TILE_LAYOUT, device=dev)
    get = lambda t: ttnn.to_torch(t).float().numpy()

    def muon_dev(th, buf, g, o, ratio):
        buf = ttnn.add(buf, ttnn.multiply(ttnn.subtract(g, buf), float(f(1.0 - MU))))
        u = ttnn.subtract(buf, ttnn.multiply(ttnn.subtract(buf, g), float(f(1.0 - MU))))
        th = ttnn.add(ttnn.multiply(th, float(f(1.0 - LR * WD_M))), ttnn.multiply(o, float(f(-LR * ratio))))
        return th, buf, u

    def muon_host(th, buf, g, o, ratio):
        buf = buf + f(1.0 - MU) * (g - buf)
        u = buf - (buf - g) * f(1.0 - MU)
        return th * f(1.0 - LR * WD_M) + f(-LR * ratio) * o, buf, u

    def adam_dev(th, m, v, g, bc1, bc2):
        m = ttnn.add(ttnn.multiply(m, B1), ttnn.multiply(g, float(1.0 - B1)))
        v = ttnn.add(ttnn.multiply(v, B2), ttnn.multiply(ttnn.multiply(g, g), float(1.0 - B2)))
        den = ttnn.add(ttnn.sqrt(ttnn.multiply(v, float(1.0 / bc2))), EPS)
        upd = ttnn.multiply(ttnn.add(ttnn.divide(ttnn.multiply(m, float(1.0 / bc1)), den), ttnn.multiply(th, WD_A)), LR)
        return ttnn.subtract(th, upd), m, v

    def adam_host(th, m, v, g, bc1, bc2):
        m = m * f(B1) + f(1.0 - B1) * g
        v = v * f(B2) + f(1.0 - B2) * (g * g)
        upd = f(LR) * ((m / f(bc1)) / (np.sqrt(v / f(bc2)) + f(EPS)) + f(WD_A) * th)
        return th - upd, m, v

    def err(d, h):
        return dict(rel=float(np.abs(d - h).max() / max(np.abs(h).max(), 1e-30)), bitid=float((d == h).mean()))

    rng = np.random.default_rng(0)
    shapes = json.loads(open(a.shapes).read().splitlines()[2])
    res = dict(host="qb2", chip=card, board="p300c", sha=sha, broken=a.broken, muon=[], adam=[])
    ok = True
    for i, ((r, c), _cnt) in enumerate(shapes[:8]):  # grade 8 shapes (all big ones) x 1 matrix each, 3 steps
        th, buf, g, o = (rng.standard_normal((r, c)).astype(f) * s for s in (0.02, 1e-3, 1e-3, 0.05))
        ratio = 0.2 * float(np.sqrt(max(r, c)))
        dt, db = put(th), put(buf)
        for step in range(3):
            g = rng.standard_normal((r, c)).astype(f) * f(1e-3)
            if a.broken and i == 0 and step == 2:  # the skipped shard: this tensor's last step never runs
                du = put(u)
            else:
                dt, db, du = muon_dev(dt, db, put(g), put(o), ratio)
            th, buf, u = muon_host(th, buf, g, o, ratio)
        dth = get(dt)
        e = dict(shape=[r, c], theta=err(dth, th), buf=err(get(db), buf), u=err(get(du), u))
        ok &= all(e[k]["rel"] <= 1e-6 for k in ("theta", "buf", "u"))
        res["muon"].append(e)
    for n in (1024, 179456 // 2):
        th, m, v = (rng.standard_normal((32, n // 32)).astype(f) * s for s in (0.1, 1e-4, 1e-8))
        v = np.abs(v)
        dth, dm, dv = put(th), put(m), put(v)
        for step in range(1, 4):
            g = rng.standard_normal(th.shape).astype(f) * f(1e-3)
            bc1, bc2 = 1 - B1 ** step, 1 - B2 ** step
            dth, dm, dv = adam_dev(dth, dm, dv, put(g), bc1, bc2)
            th, m, v = adam_host(th, m, v, g, bc1, bc2)
        e = dict(n=n, theta=err(get(dth), th), m=err(get(dm), m), v=err(get(dv), v))
        ok &= all(e[k]["rel"] <= 1e-6 for k in ("theta", "m", "v"))
        res["adam"].append(e)
    res["B2B3"] = "PASS" if ok else "FAIL"
    if a.time:
        clk, stop = [], threading.Event()

        def sampler():
            p = f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"
            while not stop.is_set():
                try:
                    clk.append((time.time(), int(open(p).read().split()[0])))
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.01)
        threading.Thread(target=sampler, daemon=True).start()
        mats = []
        for (r, c), cnt in shapes:
            z = torch.zeros(cnt, r, c)
            mats.append([ttnn.from_torch(z, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev) for _ in range(4)])
        ad = [ttnn.from_torch(torch.zeros(32, 179456 // 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
              for _ in range(4)]

        def full():
            out = [muon_dev(t, b, g_, o_, 1.0) for t, b, g_, o_ in mats]
            out.append(adam_dev(ad[0], ad[1], ad[2], ad[3], 0.1, 0.001))
            return out
        full(); ttnn.synchronize_device(dev)
        ts = []
        for _ in range(5):
            t0 = time.time(); o_ = full(); ttnn.synchronize_device(dev); ts.append((t0, time.time())); del o_
        stop.set()
        ck = [c for t, c in clk if any(t0 <= t <= t1 for t0, t1 in ts)]
        res["time"] = dict(windows=ts, step_s=[round(t1 - t0, 4) for t0, t1 in ts], aiclk=[min(ck), max(ck), len(ck)] if ck else None,
                           elements=int(sum(r * c * cnt for (r, c), cnt in shapes) + 179456))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
