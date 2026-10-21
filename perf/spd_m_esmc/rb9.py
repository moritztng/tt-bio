"""Readback of a model output, by method: what the ~5-7 ms of a page batch's to_torch is made of.

ph8 (WH, .110) put readback at 7.3 ms of 90.5 on 300M normal (to_torch 4.84 ms for 7.3 MB, 1.5 GB/s) and "rest"
at 4.4 ms. This times each way of getting a bf16 TILE DRAM tensor of the page shapes to a float32 torch tensor, and
the host-only work of embed_sequences around it, so a lever reads fewer bytes or copies once instead of guessing.

    python3 perf/spd_m_esmc/rb9.py --out rb9.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

from tt_bio.tenstorrent import download, get_device

SHAPES = {"300m": (7, 544, 960), "600m": (7, 544, 1152), "6b": (1, 544, 2560)}


def timed(fn, dev, rounds):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    ms = []
    for _ in range(rounds):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(ms), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = get_device()
    out = {}
    for name, shp in SHAPES.items():
        ref = torch.randn(shp).to(torch.bfloat16)
        x = ttnn.from_torch(ref, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        host_rm = ttnn.from_device(rm)
        r = {"mb": round(ref.numel() * 2 / 1e6, 2)}
        r["shipped"] = timed(lambda: torch.Tensor(download(x)).to(torch.float32), dev, args.rounds)
        r["untilize_dev"] = timed(lambda: ttnn.deallocate(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)), dev, args.rounds)
        r["from_device_rm"] = timed(lambda: ttnn.from_device(rm), dev, args.rounds)
        r["host_to_torch_rm"] = timed(lambda: ttnn.to_torch(host_rm), dev, args.rounds)
        h = ttnn.to_torch(host_rm)
        r["cast_to_fp32"] = timed(lambda: torch.Tensor(h).to(torch.float32), dev, args.rounds)
        r["to_torch_tile_hostuntilize"] = timed(lambda: ttnn.to_torch(x), dev, args.rounds)
        r["to_torch_rm"] = timed(lambda: ttnn.to_torch(rm), dev, args.rounds)
        try:
            dst = ttnn.allocate_tensor_on_host(rm.spec, dev)
            r["copy_d2h_prealloc"] = timed(lambda: ttnn.copy_device_to_host_tensor(rm, dst), dev, args.rounds)
            r["prealloc_to_torch"] = timed(lambda: ttnn.to_torch(dst), dev, args.rounds)
        except Exception as exc:  # noqa: BLE001 - a missing API is a result here, not a crash
            r["copy_d2h_prealloc"] = repr(exc)[:200]
        try:
            f32 = ttnn.typecast(rm, ttnn.float32)
            r["to_torch_rm_fp32dev"] = timed(lambda: ttnn.to_torch(f32), dev, args.rounds)
            ttnn.deallocate(f32)
        except Exception as exc:  # noqa: BLE001
            r["to_torch_rm_fp32dev"] = repr(exc)[:200]
        got = torch.Tensor(download(x)).to(torch.float32)
        r["exact"] = bool(torch.equal(got, ref.float()))
        # host-only post of embed_sequences on this output: per-row slice, numpy, float32, mean pool
        lens = [shp[1] - 30] * shp[0]
        def post(copy):
            for i, li in enumerate(lens):
                e = got[i, :li].numpy()
                e = e.astype(np.float32) if copy else e
                e[1:-1].mean(axis=0).astype(np.float32)
        r["post_shipped"] = timed(lambda: post(True), dev, args.rounds)
        r["post_nocopy"] = timed(lambda: post(False), dev, args.rounds)
        for t in (x, rm):
            ttnn.deallocate(t)
        out[name] = r
        print(name, json.dumps(r), flush=True)
    from tt_bio import esmc
    seq = Path("scripts/gpu_vs_tt/fixtures/prot512.seq").read_text().strip()
    out["batch_tokens_7"] = timed(lambda: esmc._batch_tokens([seq] * 7), dev, args.rounds)
    out["batch_tokens_7_ragged"] = timed(lambda: esmc._batch_tokens([seq[:-i] for i in range(1, 8)]), dev, args.rounds)
    # the eager forward's uploads, in call order (forward -> _dispatch -> _forward_eager -> module)
    ids, lens, mask, kv = esmc._batch_tokens([seq[:-i] for i in range(7)])
    synced = lambda f: (lambda: (f(), ttnn.synchronize_device(dev)))
    out["key_lens"] = timed(lambda: esmc._key_lens(mask), dev, args.rounds)
    kl = esmc._key_lens(mask)
    out["up_tokens"] = timed(synced(lambda: ttnn.from_torch(ids.to(torch.int32), device=dev, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                           dtype=ttnn.uint32)), dev, args.rounds)
    out["up_keypad"] = timed(synced(lambda: esmc._lm_mask(mask, kl, dev)), dev, args.rounds)
    out["up_kv"] = timed(synced(lambda: ttnn.from_torch(kv.to(torch.bfloat16), device=dev, layout=ttnn.TILE_LAYOUT,
                                                       dtype=ttnn.bfloat16)), dev, args.rounds)
    for hd in (64, 128):
        out[f"rope_tables_544_{hd}"] = timed(synced(lambda: esmc.rope_tables(544, hd, device=dev)), dev, args.rounds)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
