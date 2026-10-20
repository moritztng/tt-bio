#!/usr/bin/env python3
"""bc4-bet-layout: a slice's gradient padded back with zeros, `autograd._pad_slice`'s concat against one
`ttnn.pad`, at the shapes BindCraft 2's census shows. Bit-equality and median ms of 7 bursts of 20.
    python perf/bc4_layout/padslice_ab.py out.json
"""
import json, os, pathlib, statistics, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# (cut shape, full shape): the template triatt d=16 head, and the MSA qkv thirds.
CASES = [((224, 4, 224, 16), (224, 4, 224, 32)),
         ((2, 224, 256), (2, 224, 768)),
         ((1, 224, 256), (2, 224, 256))]


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    dev = T.get_device()
    card = int((os.environ.get("TT_VISIBLE_DEVICES") or "0").split(",")[0])
    aiclk = lambda: int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read())
    res = {"card": card, "cases": []}
    for cut, full in CASES:
        torch.manual_seed(0)
        g = ttnn.from_torch(torch.randn(cut).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        ends = list(cut)
        arms = {"concat": lambda: ag._pad_slice(g, [0] * len(cut), ends, full),
                "pad": lambda: ttnn.pad(g, [(0, f - c) for c, f in zip(cut, full)], 0.0)}
        outs = {k: ttnn.to_torch(f()) for k, f in arms.items()}
        row = {"cut": cut, "full": full, "bit_equal": bool(torch.equal(outs["concat"], outs["pad"])),
               "shapes": {k: list(v.shape) for k, v in outs.items()}}
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
        row["aiclk_min_max"] = [min(clk), max(clk)]
        res["cases"].append(row)
        print(json.dumps(row), flush=True)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
