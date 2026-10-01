#!/usr/bin/env python3
"""C10 at the op: is adding/dropping a leading axis a VIEW or a DRAM move?

The attend path pays 0.2548 GB a block for `ttnn.unsqueeze(qkv_in, 1)` and 0.0849 GB for
`ttnn.squeeze(o_heads, 1)`. If `ttnn.reshape` to the same shape is a view at tile layout, both
are a one-line deletion. Equality first, then synced walls, arms alternated, AICLK during."""
import json, pathlib, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    card, out = sys.argv[1], sys.argv[2]
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    clk = pathlib.Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    torch.manual_seed(0)
    res = {"card": card, "aiclk": []}

    def sample():
        try:
            res["aiclk"].append(int(clk.read_text().strip()))
        except Exception:
            pass

    # qkv_in as `attend` sees it: the triangle-attention projection at n=288, 8 heads x 32.
    cases = {
        "unsqueeze_qkv": ((288, 288, 768), 1),   # tenstorrent.py:9214, 0.2548 GB / 2 calls
        "squeeze_out":   ((288, 1, 288, 128), 1),  # the o_heads drop at :9316, 0.0849 GB / 2
    }
    for name, (shape, axis) in cases.items():
        t = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev)
        drop = name.startswith("squeeze")
        tgt = list(shape)
        if drop:
            tgt.pop(axis)
        else:
            tgt.insert(axis, 1)
        tgt = tuple(tgt)

        def arm_native():
            return ttnn.squeeze(t, axis) if drop else ttnn.unsqueeze(t, axis)

        def arm_reshape():
            return ttnn.reshape(t, tgt)

        a, b = arm_native(), arm_reshape()
        same = torch.equal(ttnn.to_torch(a).float(), ttnn.to_torch(b).float())
        # A view shares the device buffer address with its source; a move allocates a new one.
        addr = {}
        for k, v in (("src", t), ("native", a), ("reshape", b)):
            try:
                addr[k] = v.buffer_address()
            except Exception as e:
                addr[k] = f"n/a ({type(e).__name__})"
        ttnn.deallocate(a); ttnn.deallocate(b)
        walls = {"native": [], "reshape": []}
        for rep in range(7):
            for label, fn in (("native", arm_native), ("reshape", arm_reshape)) if rep % 2 == 0 \
                    else (("reshape", arm_reshape), ("native", arm_native)):
                ttnn.synchronize_device(dev)
                sample()
                t0 = time.perf_counter()
                r = fn()
                ttnn.synchronize_device(dev)
                walls[label].append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(r)
        med = lambda xs: sorted(xs)[len(xs) // 2]
        res[name] = {"shape": list(shape), "target": list(tgt), "equal": same,
                     "buffer_address": addr,
                     "native_ms": round(med(walls["native"][1:]), 4),
                     "reshape_ms": round(med(walls["reshape"][1:]), 4),
                     "walls": {k: [round(x, 4) for x in v] for k, v in walls.items()}}
        ttnn.deallocate(t)
        r = res[name]
        print(f"{name:16s} equal={r['equal']}  native {r['native_ms']:.4f} ms  "
              f"reshape {r['reshape_ms']:.4f} ms  addr={r['buffer_address']}", flush=True)
    a = res["aiclk"]
    res["aiclk_median"] = sorted(a)[len(a) // 2] if a else None
    res["aiclk_min"] = min(a) if a else None
    res["aiclk_n"] = len(a)
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    print(f"AICLK median {res['aiclk_median']} min {res['aiclk_min']} n {res['aiclk_n']}")


if __name__ == "__main__":
    main()
