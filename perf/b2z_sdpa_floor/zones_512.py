#!/usr/bin/env python3
"""Per-stage device time of the triangle-attention SDPA at [B, H, 512, 32], shipped levers.

ablate_512 (sab, WH, 10-18) found the exp at 4 % of the op and the matmul fidelity at 4 %: the other
~92 % is invisible to both roofs. This reads it directly. The kernel is compiled with TRIATT_ZONES
(per-stage profiler zones, values unchanged) and run under a profiler build of tt-metal
(TT_METAL_DEVICE_PROFILER=1, perf/spd_census/run.sh's environment). Stages per (batch, head, q chunk,
k chunk) unit: TA_QK (Q K^T with the mask preloaded into DST, including the wait on K), TA_MAX (row
max), TA_SUBEXP (exp(s - max) with the partial row sum), TA_PV (P V, including the wait on V),
TA_CORR (online-softmax rescale, k chunks after the first).

    TT_BIO_TRIATT_ZONES=1 TT_METAL_DEVICE_PROFILER=1 python perf/b2z_sdpa_floor/zones_512.py --heads 4

`--batch` below 512 keeps every core's marker buffer from overflowing (a unit's stage times do not
depend on how many units a core runs, only DRAM contention does; compare the two).
"""
import argparse, collections, contextlib, csv, json, os, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

S, D = 512, 32
STAGES = ("TA_QK", "TA_MAX", "TA_SUBEXP", "TA_PV", "TA_CORR")


def find_csv():
    roots = [os.environ.get("TT_METAL_PROFILER_DIR"), os.path.join(os.environ.get("TT_METAL_HOME", ""),
                                                                    "generated", "profiler")]
    # os.walk, not glob: the profiler writes under a hidden `.logs`, which a recursive glob skips
    hits = [os.path.join(d, "profile_log_device.csv") for r in roots if r and os.path.isdir(r)
            for d, _, fs in os.walk(r) if "profile_log_device.csv" in fs]
    if not hits:
        raise SystemExit(f"no profile_log_device.csv under {roots}")
    return max(hits, key=os.path.getmtime)


def parse(path):
    """{risc: {stage: [cycles per unit]}}, the chip MHz, and per-core unit counts."""
    with open(path) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0]) if "CHIP_FREQ" in head else None
        rows = csv.reader(f)
        cols = [c.strip() for c in next(rows)]
        ix = {c: cols.index(c) for c in cols}
        t, risc, zone, typ = (ix["time[cycles since reset]"], ix["RISC processor type"], ix["zone name"],
                              ix["type"])
        cx, cy = ix["core_x"], ix["core_y"]
        open_ = {}
        out = collections.defaultdict(lambda: collections.defaultdict(list))
        units = collections.Counter()
        for r in rows:
            z = r[zone].strip()
            if z not in STAGES:
                continue
            key = (r[cx].strip(), r[cy].strip(), r[risc].strip(), z)
            if r[typ].strip() == "ZONE_START":
                open_[key] = int(r[t])
            elif key in open_:
                out[key[2]][z].append(int(r[t]) - open_.pop(key))
                if z == "TA_QK":
                    units[(key[0], key[1], key[2])] += 1
    return out, mhz, units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=4, help="Boltz-2 4, Protenix-v2 8")
    ap.add_argument("--batch", type=int, default=S)
    ap.add_argument("--calls", type=int, default=1, help="profiled calls after the warm-up")
    ap.add_argument("--b8", action="store_true", help="q, k, v and bias bfp8, as triatt_ln_b8 feeds the op (WH --fast)")
    ap.add_argument("--levers", default=None, help="lever set to run the op under (default: the process default)")
    ap.add_argument("--out", default="perf/b2z_sdpa_floor/zones_512.json")
    args = ap.parse_args()
    assert os.environ.get("TT_BIO_TRIATT_ZONES"), "set TT_BIO_TRIATT_ZONES=1 (and run under a profiler build)"
    import torch
    import ttnn
    from tt_bio import tenstorrent as T

    dev = T.get_device()
    B, H = args.batch, args.heads

    def dram(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat8_b if args.b8 else ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    q, k, v = (dram(torch.randn(B, H, S, D).to(torch.bfloat16)) for _ in range(3))
    bias = dram(torch.randn(1, H, S, S).to(torch.bfloat16))
    scale = D ** -0.5

    def call():
        with T.levers(args.levers) if args.levers else contextlib.nullcontext():
            o = T._tri_att_sdpa_at(q, k, v, bias, scale)
        assert o is not None, "triatt SDPA declined this shape"
        return o

    for _ in range(2):
        ttnn.deallocate(call())
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)  # flush the warm-up, so the log's last calls are the timed ones
    t0 = time.perf_counter()
    for _ in range(args.calls):
        ttnn.deallocate(call())
    ttnn.synchronize_device(dev)
    wall_ms = (time.perf_counter() - t0) * 1e3 / args.calls
    ttnn.ReadDeviceProfiler(dev)

    path = find_csv()
    per, mhz, units = parse(path)
    mhz = mhz or 1000.0
    res = {"doc": __doc__, "meta": {"card": os.environ.get("TT_VISIBLE_DEVICES"), "shape": [B, H, S, D],
                                    "b8": args.b8, "levers": args.levers,
                                    "grid": list(T.COMPUTE_GRID_MAIN), "mhz": mhz, "csv": path,
                                    "wall_ms_per_call": wall_ms, "loadavg": os.getloadavg(),
                                    "units_per_core": dict(collections.Counter(units.values()))},
           "stages": {}}
    for risc in sorted(per):
        tot = sum(statistics.median(v) for v in per[risc].values())
        res["stages"][risc] = {z: {"n": len(v), "med_us": statistics.median(v) / mhz,
                                   "p90_us": sorted(v)[int(0.9 * (len(v) - 1))] / mhz,
                                   "share": statistics.median(v) / tot}
                               for z, v in per[risc].items()}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    for risc, st in res["stages"].items():
        print(risc, "  ".join(f"{z} {s['med_us']:.1f}us {s['share']:.0%} (n={s['n']})" for z, s in st.items()))
    print("units/core", res["meta"]["units_per_core"], "wall ms/call", round(wall_ms, 3))


if __name__ == "__main__":
    main()
