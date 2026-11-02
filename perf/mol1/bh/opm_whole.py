#!/usr/bin/env python3
"""mol1-bh-prof: Mol-1's MSA-module OPM at the 1024 shape, by row-block height: main's 128 rows against taller blocks.

Same process, same operands, arms interleaved. x arrives as the fold hands it: 512-row depth chunks of
[1, rows, N, c_m], plus the cycle's [depth, 1, 1] row mask, and a [1, N, N, c_z] residual. Lever set
NORMAL | WIDE_NORMAL | MOL1_BH (Mol-1 normal mode on Blackhole). `whole` sets OPM_WHOLE_DRAM_DEN so the
z = I * C * D * J * 2 byte arm does not block; an integer arm sets the block height (OPM_Z_BUDGET_BYTES and
OPM_CHUNK_SIZE). Reports per-call ms (median of reps, one sync per call),
bit-equality of the two outputs, the row census and AICLK of every card sampled during the timed calls.

    TT_VISIBLE_DEVICES=<card> timeout 1200 python opm_whole.py out.json [--depths 2048,6144] [--n 1024]
"""
import argparse, glob, json, os, pathlib, statistics, sys, threading, time
import torch

ROOT = pathlib.Path(os.environ.get("TT_BIO_ROOT", pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--depths", default="2048,6144")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--arms", default="blocked,whole", help="blocked (main), whole, or a block height in rows")
    A = p.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as TT
    from tt_bio import protenix as P
    dev = TT.get_device()
    cfg = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    lv = P.NORMAL_LEVERS | P.WIDE_NORMAL | P.MOL1_BH
    N, cm, C, cz = A.n, 192, 32, 448
    g = torch.Generator().manual_seed(0)
    sd = {"norm.weight": 1 + 0.1 * torch.randn(cm, generator=g), "norm.bias": 0.1 * torch.randn(cm, generator=g),
          "proj_a.weight": torch.randn(C, cm, generator=g) * cm ** -0.5,
          "proj_b.weight": torch.randn(C, cm, generator=g) * cm ** -0.5,
          "proj_o.weight": torch.randn(cz, C * C, generator=g) * (C * C) ** -0.5,
          "proj_o.bias": 0.1 * torch.randn(cz, generator=g)}
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    clk_paths = sorted(glob.glob("/sys/class/tenstorrent/*/tt_aiclk"))
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            samples.append([int(open(f).read()) for f in clk_paths])
            time.sleep(0.2)
    th = threading.Thread(target=poll, daemon=True)
    res = {"host": os.uname().nodename, "sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
           "levers": sorted(lv), "clk_paths": clk_paths, "depths": {}}
    den0, bud0, cap0 = TT.OPM_WHOLE_DRAM_DEN, TT.OPM_Z_BUDGET_BYTES, TT.OPM_CHUNK_SIZE
    names = A.arms.split(",")

    def arm_set(arm):
        TT.OPM_WHOLE_DRAM_DEN, TT.OPM_Z_BUDGET_BYTES, TT.OPM_CHUNK_SIZE = den0, bud0, cap0
        if arm == "whole":
            TT.OPM_WHOLE_DRAM_DEN = 1
        elif arm != "blocked":
            TT.OPM_Z_BUDGET_BYTES = int(arm) * 32 * 32 * N * 2
            TT.OPM_CHUNK_SIZE = int(arm)
    with TT.levers(lv):
        opm = TT.OuterProductMean(sd, cfg, scale_bias=True)
        th.start()
        for depth in (int(d) for d in A.depths.split(",")):
            chunks = [up(torch.randn(1, min(512, depth - s), N, cm, generator=g)) for s in range(0, depth, 512)]
            mask = up(torch.ones(depth, 1, 1))
            res0 = torch.randn(1, N, N, cz, generator=g).to(torch.bfloat16)
            row, outs = {}, {}
            for arm in names:
                row[arm] = {"ms": []}
            for rep in range(A.reps + 1):
                for arm in names:
                    arm_set(arm)
                    z = up(res0)
                    stats0 = dict(TT.OPM_ROW_STATS)
                    ttnn.synchronize_device(dev)
                    n0 = len(samples)
                    t0 = time.perf_counter()
                    out = opm(list(chunks), mask, None, residual=z)
                    ttnn.synchronize_device(dev)
                    ms = (time.perf_counter() - t0) * 1e3
                    if rep == 0:
                        outs[arm] = ttnn.to_torch(out)
                        row[arm]["census"] = {k: TT.OPM_ROW_STATS[k] - stats0[k] for k in stats0}
                        row[arm]["cold_ms"] = ms
                    else:
                        row[arm]["ms"].append(ms)
                        row[arm].setdefault("clk", []).extend(samples[n0:])
                    for t in {id(out): out, id(z): z}.values():
                        try:
                            ttnn.deallocate(t)
                        except Exception:
                            pass
            arm_set("blocked")
            for arm in names:
                clk = row[arm].pop("clk", [])
                row[arm]["ms_median"] = statistics.median(row[arm]["ms"])
                row[arm]["aiclk_max_card"] = [max(c[i] for c in clk) for i in range(len(clk_paths))] if clk else None
                row[arm]["aiclk_min_card"] = [min(c[i] for c in clk) for i in range(len(clk_paths))] if clk else None
            base = outs[names[0]]
            row["bit_equal_first"] = {a: bool(torch.equal(outs[a], base)) for a in names}
            fl = 2.0 * (depth * N * cm * 64 + depth * N * N * C * C + N * N * C * C * cz)
            for arm in names:
                row[arm]["tflops"] = fl / row[arm]["ms_median"] / 1e9
            res["depths"][str(depth)] = row
            print(depth, {a: (round(row[a]["ms_median"], 1), round(row[a]["tflops"], 1), row[a]["census"],
                              row[a]["aiclk_min_card"]) for a in names},
                  "bit_equal", row["bit_equal_first"], flush=True)
            for t in chunks + [mask]:
                ttnn.deallocate(t)
        stop.set()
    pathlib.Path(A.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
