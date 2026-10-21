#!/usr/bin/env python3
"""Triangle-attention backward (`triatt_bw.run`) per call: each arm graded against float64 and timed.

Arms are `name:ATTR=VAL,ATTR=VAL` on the `tt_bio.triatt_bw` module, separated by `;`; `v2=0/1` is
`run`'s argument instead, forcing the whole-query body (unset: the board's serving plan). Every arm runs
on the same bf16 q, k, v, bias and cotangent; the float64 reference sees those same bf16 values, so
what is graded is the kernel's arithmetic. Timing is synced per call, arms alternated every rep.

    TT_VISIBLE_DEVICES=<chip> python perf/bc4_triatt/bw_bench.py --n 224 \
        --arms "plain:EXP_21F=1,v2=0;v2:EXP_21F=1,v2=1"

With TT_BIO_TRIATT_BW_ZONES=1 under a device-profiler build, `--zones` reads the per-phase zones of
one profiled call per arm instead of timing (math RISC, median over core-units, us at the CSV's MHz).
"""
import argparse, collections, csv, json, os, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_arms(spec):
    arms = []
    for part in filter(None, (s.strip() for s in spec.split(";"))):
        name, _, kv = part.partition(":")
        attrs = {}
        for item in filter(None, kv.split(",")):
            k, _, v = item.partition("=")
            attrs[k] = {"1": True, "0": False}.get(v, v)
        arms.append((name, attrs))
    return arms


def zones(path, mhz=None):
    """{zone: median us} over every (core, unit) on the math RISC, from profile_log_device.csv."""
    with open(path) as f:
        head = f.readline()
        if "CHIP_FREQ" in head:
            mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = csv.reader(f)
        cols = [c.strip() for c in next(rows)]
        ix = {c: cols.index(c) for c in cols}
        open_, out = {}, collections.defaultdict(list)
        for r in rows:
            if r[ix["RISC processor type"]].strip() != "TRISC_1":
                continue
            z = r[ix["zone name"]].strip()
            key = (r[ix["core_x"]], r[ix["core_y"]], z)
            if r[ix["type"]].strip() == "ZONE_START":
                open_[key] = int(r[ix["time[cycles since reset]"]])
            elif r[ix["type"]].strip() == "ZONE_END" and key in open_:
                out[z].append(int(r[ix["time[cycles since reset]"]]) - open_.pop(key))
    mhz = mhz or 1350.0
    return {z: {"n": len(v), "med_us": statistics.median(v) / mhz} for z, v in out.items()}, mhz


def find_csv():
    root = os.environ.get("TT_METAL_PROFILER_DIR") or os.path.join(os.environ.get("TT_METAL_HOME", ""),
                                                                    "generated", "profiler")
    hits = [os.path.join(d, "profile_log_device.csv") for d, _, fs in os.walk(root)
            if "profile_log_device.csv" in fs]
    return max(hits, key=os.path.getmtime) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=224)
    ap.add_argument("--lead", type=int, default=0, help="leading axis, default n")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--bias-std", type=float, default=0.5)
    ap.add_argument("--arms", default="base:EXP_21F=1")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--zones", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import triatt_bw as TB
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    B, H, N, D = a.lead or a.n, a.heads, a.n, 32
    scale = D ** -0.5
    torch.manual_seed(0)
    q, k, v, g = (torch.randn(B, H, N, D).to(torch.bfloat16) for _ in range(4))
    bias = (torch.randn(1, H, N, N) * a.bias_std).to(torch.bfloat16)
    qr, kr, vr, br = (t.double().requires_grad_(True) for t in (q, k, v, bias))
    (torch.softmax(qr @ kr.transpose(-1, -2) * scale + br, dim=-1) @ vr).backward(g.double())
    ref = {"dq": qr.grad, "dk": kr.grad, "dv": vr.grad, "dbias": br.grad}

    up = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,  # noqa: E731
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tq, tk, tv, tg, tb = (up(x) for x in (q, k, v, g, bias))
    arms = parse_arms(a.arms)
    defaults = {n: getattr(TB, n) for _, at in arms for n in at if n != "v2"}

    def call(attrs):
        for n, val in defaults.items():
            setattr(TB, n, attrs.get(n, val))
        return TB.run(dev, tq, tk, tv, tb, tg, scale, (ttnn.MathFidelity.HiFi4,), v2=attrs.get("v2"))

    res = {"shape": [B, H, N, D], "bias_std": a.bias_std, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "plan": {k_: v_ for k_, v_ in TB.serving_plan(B, H, N, D, (dev.compute_with_storage_grid_size().x,
                                                                      dev.compute_with_storage_grid_size().y)).items()},
           "arms": {}}
    for name, attrs in arms:
        out = [torch.Tensor(ttnn.to_torch(t)).double() for t in call(attrs)]
        grade = {}
        for key, got in zip(("dq", "dk", "dv", "dbias"), out):
            r = ref[key]
            got = got.reshape(r.shape)
            grade[key] = float((got - r).norm() / r.norm())
        res["arms"][name] = {"attrs": {k_: str(v_) for k_, v_ in attrs.items()}, "rel_l2_vs_f64": grade}
        print(name, json.dumps(grade), flush=True)

    if a.zones:
        for name, attrs in arms:
            for x in call(attrs):
                ttnn.deallocate(x)
            ttnn.synchronize_device(dev)
            ttnn.ReadDeviceProfiler(dev)
            for x in call(attrs):
                ttnn.deallocate(x)
            ttnn.synchronize_device(dev)
            ttnn.ReadDeviceProfiler(dev)
            path = find_csv()
            if path:
                z, mhz = zones(path)
                res["arms"][name]["zones"] = z
                res["arms"][name]["zones_mhz"] = mhz
                print(name, " ".join(f"{k_} {v_['med_us']:.1f}" for k_, v_ in z.items()), flush=True)
    else:
        t = {name: [] for name, _ in arms}
        for x in call(arms[0][1]):
            ttnn.deallocate(x)
        for rep in range(a.reps):
            for name, attrs in (arms if rep % 2 == 0 else arms[::-1]):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                r = call(attrs)
                ttnn.synchronize_device(dev)
                t[name].append(time.perf_counter() - t0)
                for x in r:
                    ttnn.deallocate(x)
        for name, _ in arms:
            ts = sorted(t[name])
            res["arms"][name].update(median_ms=1e3 * ts[len(ts) // 2], min_ms=1e3 * ts[0],
                                     p90_ms=1e3 * ts[int(0.9 * (len(ts) - 1))], n=len(ts))
            print(name, f"median {1e3 * ts[len(ts) // 2]:.3f} ms  min {1e3 * ts[0]:.3f}", flush=True)
    res["loadavg"] = os.getloadavg()
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
