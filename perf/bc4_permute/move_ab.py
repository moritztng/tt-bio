#!/usr/bin/env python3
"""bc4-bet-permute: the trimul's back move, permute(0,2,3,1) of [1, C, N, N] bf16 TILE, by every route
tt-bio has, at BindCraft 2's shapes. Bit-equality against torch and median ms of 7 bursts of 20.
    python perf/bc4_permute/move_ab.py out.json [--n 224]
"""
import json, os, pathlib, statistics, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as T, reblock_permute as R, permute_back_pk as PK
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 224
    dev = T.get_device()
    card = int((os.environ.get("TT_VISIBLE_DEVICES") or "0").split(",")[0])
    aiclk = lambda: int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read())
    dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    res = {"card": card, "n": n, "cases": []}
    for c in (128, 64):
        for src_name, src in (("dram", dram), ("l1", l1)):
            torch.manual_seed(0)
            xt = torch.randn(1, c, n, n).to(torch.bfloat16)
            x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=src)
            ref = xt.permute(0, 2, 3, 1)

            def pbpk():
                out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, n, c]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, dram)
                return PK.permute_back_pk(x, out, tuple(T.COMPUTE_GRID_MAIN))

            def two():
                a = ttnn.transpose(x, 1, 2, memory_config=dram)
                b = ttnn.transpose(a, 2, 3, memory_config=dram)
                ttnn.deallocate(a)
                return b

            arms = {"stock_permute": lambda: ttnn.permute(x, (0, 2, 3, 1), memory_config=dram),
                    "two_transposes": two, "pbpk": pbpk,
                    "tt_bio_back": lambda: R.reblock_permute_back(x, dram)}
            row = {"c": c, "n": n, "src": src_name,
                   "eligible_back": bool(R.eligible_back(x, dram)), "bit_equal": {}}
            for k, f in list(arms.items()):
                try:
                    row["bit_equal"][k] = bool(torch.equal(ttnn.to_torch(f()), ref))
                except Exception as e:
                    print(k, "raised", repr(e)[:300], flush=True)
                    arms.pop(k)
            ms = {k: [] for k in arms}
            clk = []
            for _ in range(7):
                for k, f in arms.items():
                    ttnn.synchronize_device(dev); t0 = time.perf_counter()
                    for _ in range(20):
                        ttnn.deallocate(f())
                    ttnn.synchronize_device(dev); ms[k].append((time.perf_counter() - t0) * 1e3 / 20)
                    clk.append(aiclk())
            for k in arms:
                row[k] = {"ms": round(statistics.median(ms[k]), 4), "spread_ms": round(max(ms[k]) - min(ms[k]), 4)}
            row["aiclk_min_med_max"] = [min(clk), int(statistics.median(clk)), max(clk)]
            res["cases"].append(row)
            print(json.dumps(row), flush=True)
            ttnn.deallocate(x)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
