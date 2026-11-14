#!/usr/bin/env python3
"""mol1-bh-diffstep: the token DiT's fp32 attention (`_sdpa32_fused`, Mol-1's dit_qkvf path) with and without
`dit_qbox`, at the padded token counts of the 256/512/1024 inputs, 5 samples, 16 heads, head lanes 64.

Per shape and arm: device ms per call by trace replay (>= 30 ms of device time), the output compared bit for bit
between arms, max |diff| vs a float64 reference on sample 0, and AICLK sampled during the replays.

    TT_VISIBLE_DEVICES=<card> timeout 900 python sdpa_box_ab.py out.json
"""
import glob, json, math, os, pathlib, sys, threading, time
import torch

ROOT = pathlib.Path(os.environ.get("TT_BIO_ROOT", pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
S, H, DH, DREAL = 5, 16, 64, 48


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as TT
    dev = TT.get_device()
    clk = sorted(glob.glob("/sys/class/tenstorrent/*/tt_aiclk"))
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            samples.append([int(open(f).read()) for f in clk])
            time.sleep(0.1)
    threading.Thread(target=poll, daemon=True).start()

    def replay(fn):
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out = fn()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        n = 5
        while True:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) * 1000
            if dt >= 30 or n >= 400:
                break
            n = min(400, 2 * n)
        ttnn.release_trace(dev, tid)
        ttnn.deallocate(out)
        return dt / n

    res = {"sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(), "shapes": {}}
    g = torch.Generator().manual_seed(0)
    base = set(TT._LEVERS) | {"dit_sdpa32", "dit_qkvf"}
    for n in (256, 512, 1024):
        # q, k, v heads in the first DREAL of each 64-wide lane (the padded head lanes are zero, as apb_cat lays them)
        qkv = torch.zeros(S, 1, n, 3 * H * DH)
        qkv.view(S, 1, n, 3 * H, DH)[..., :DREAL] = torch.randn(S, 1, n, 3 * H, DREAL, generator=g)
        bias = torch.randn(1, H, n, n, generator=g) * 2
        scale = 1 / math.sqrt(DREAL)
        tq = ttnn.from_torch(qkv, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        tm = ttnn.from_torch(bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        q, k, v = (qkv.view(S, n, 3, H, DH)[:, :, i].permute(0, 2, 1, 3).double() for i in range(3))
        ref = torch.softmax((q @ k.transpose(-1, -2) + bias.double()) * scale, -1) @ v   # [S, H, n, DH]
        ref = ref.permute(0, 2, 1, 3).reshape(S, n, H * DH)
        row = {}
        outs = {}
        for arm, lev in (("base", base - {"dit_qbox"}), ("qbox", base | {"dit_qbox"}),
                         ("base2", base - {"dit_qbox"}), ("qbox2", base | {"dit_qbox"})):
            TT._LEVERS = frozenset(lev)
            ch = TT._sdpa32_chunks(dev, S, H, n)
            fn = lambda: TT._sdpa32_fused(tq, tm, scale, H)  # noqa: E731
            o = fn()
            outs[arm] = ttnn.to_torch(o).reshape(S, n, H * DH)
            ttnn.deallocate(o)
            i0 = len(samples)
            ms = replay(fn)
            c = [x for s in samples[i0:] for x in s]
            row[arm] = {"ms": round(ms, 4), "q_chunk": ch[0], "boxes": len(ch[3]) if ch[3] else None,
                        "aiclk": [min(c), max(c)] if c else None,
                        "max_err_vs_f64": float((outs[arm].double() - ref).abs().max())}
        row["bit_exact"] = bool(torch.equal(outs["base"], outs["qbox"]))
        res["shapes"][n] = row
        print(n, json.dumps(row), flush=True)
        ttnn.deallocate(tq); ttnn.deallocate(tm)
    stop.set()
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
