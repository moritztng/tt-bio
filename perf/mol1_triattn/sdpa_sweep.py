#!/usr/bin/env python3
"""Triangle attention's SDPA at Mol-1's widths: the served pick against every fused (q_chunk, k_chunk).

The operands are the ones the served `TriangleAttention` hands `_tri_att_sdpa` (captured on one call under
bench.py's `served` levers), so dtypes, scale and compute config match the fold. The served arm replays
`_tri_att_sdpa` and reports the pair it took (`SDPA_CHUNK_PICKS`); each candidate calls the fused kernel at an
explicit pair, `padded_mask` where the q chunk does not divide. Per cell: ms (min / median over reps), the core
split the kernel used, whether the output is bit-equal to the served one, and rel_rms against an fp32 torch SDPA
on two batch rows. AICLK sampled during the timed calls.

Usage: sdpa_sweep.py --sizes 512,1024 --widths 320x10,448x14 --qt 4,5,6,7,8,10,11,12,16 --kt 8,16,32 --out sweep.jsonl
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import triatt_sdpa as TS
import clocksample
from bench import ARMS, HD, weights

assert Path(T.__file__).resolve().is_relative_to(ROOT)


def capture(dev, S, c, H):
    """(q, k, v, bias, scale, ckc) of one served `_tri_att_sdpa` call, kept alive on device."""
    sd = weights(c, H, 7)
    torch.manual_seed(S + c)
    x = torch.randn(S, S, c).bfloat16()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    got, orig = [], T._tri_att_sdpa

    def grab(q, k, v, bias, scale, ckc=None, pad=False):
        if not got:
            got.append((ttnn.clone(q), ttnn.clone(k), ttnn.clone(v), ttnn.clone(bias), scale, ckc, pad))
        return orig(q, k, v, bias, scale, ckc, pad)
    T._MM_DEFAULT_WIDE = ARMS["served"][1]
    T._tri_att_sdpa = grab
    try:
        with T.levers(ARMS["served"][0]):
            mod = T.TriangleAttention(HD, H, False, sd, ckc, softmax_site="pairformer")
            xt = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(mod(xt))
            ttnn.deallocate(xt)
    finally:
        T._tri_att_sdpa = orig
    return got[0]


def reference(q, k, v, bias, scale, rows):
    """fp32 softmax attention of the same device operands on batch rows `rows`: [rows, H, S, d]."""
    qt, kt, vt = (ttnn.to_torch(t)[rows].float() for t in (q, k, v))
    b = ttnn.to_torch(bias).float()
    a = torch.softmax(qt @ kt.transpose(-1, -2) * scale + b, -1)
    return a @ vt


def rel(ref, got):
    return round(((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(), 6)


def timed(fn, dev, reps):
    o = fn()
    if o is None:
        return None, None, None
    ttnn.synchronize_device(dev)
    ts = []
    with clocksample.during(period=0.5) as clk:
        for _ in range(reps):
            t0 = time.perf_counter()
            r = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(r)
    ts.sort()
    return o, ts, clk.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,1024")
    ap.add_argument("--widths", default="320x10,448x14")
    ap.add_argument("--qt", default="4,5,6,7,8,10,11,12,16", help="q chunk widths in tiles")
    ap.add_argument("--kt", default="8,16,32", help="k chunk widths in tiles (dividing only)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = T.get_device()
    cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
    emit = lambda rec: (print(json.dumps(rec), flush=True), a.out.open("a").write(json.dumps(rec) + "\n"))  # noqa: E731
    for S in map(int, a.sizes.split(",")):
        for wh in a.widths.split(","):
            c, H = map(int, wh.split("x"))
            q, k, v, bias, scale, ckc, pad = capture(dev, S, c, H)
            base = {"S": S, "c_z": c, "heads": H, "grid": list(T.COMPUTE_GRID_MAIN),
                    "dtypes": [str(t.dtype) for t in (q, k, v, bias)], "pad": pad}
            rows = [0, S // 2]
            ref = reference(q, k, v, bias, scale, rows)
            T.SDPA_CHUNK_PICKS.clear()
            with T.levers(ARMS["served"][0]):
                o, ts, clk = timed(lambda: T._tri_att_sdpa(q, k, v, bias, scale, ckc, pad), dev, a.reps)
            served = ttnn.to_torch(o)
            ttnn.deallocate(o)
            pick = T.SDPA_CHUNK_PICKS.get((S, S))
            emit({**base, "arm": "served", "pick": pick, "ms_min": round(ts[0] * 1e3, 3),
                  "ms_med": round(ts[len(ts) // 2] * 1e3, 3), "aiclk": clk,
                  "rel_rms_fp32": rel(ref, served[rows].float())})
            for kt in map(int, a.kt.split(",")):
                kc = kt * 32
                if S % kc:
                    continue
                for qt in map(int, a.qt.split(",")):
                    qc = qt * 32
                    if qc > S:
                        continue
                    padded = S % qc != 0
                    qnc = -(-S // qc)
                    rec = {**base, "arm": "fused", "q_chunk": qc, "k_chunk": kc, "padded": padded,
                           "split": [cores // (H * qnc), H, qnc] if cores // (H * qnc) else None}
                    TS.REJECTS.clear()
                    try:
                        with T.levers(ARMS["served"][0]):
                            o, ts, clk = timed(lambda: TS.sdpa(q, k, v, bias, scale, qc, kc, ckc_default=ckc,
                                                               q_split_cap=0, padded_mask=padded), dev, a.reps)
                        if o is None:
                            rec["declined"] = sorted({r for r, _ in TS.REJECTS})
                        else:
                            got = ttnn.to_torch(o)
                            ttnn.deallocate(o)
                            rec.update(ms_min=round(ts[0] * 1e3, 3), ms_med=round(ts[len(ts) // 2] * 1e3, 3),
                                       aiclk=clk, equal_served=bool(torch.equal(got, served)),
                                       rel_rms_fp32=rel(ref, got[rows].float()))
                    except Exception as e:  # noqa: BLE001
                        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
                    emit(rec)
            for t in (q, k, v, bias):
                ttnn.deallocate(t)
    T.cleanup()


if __name__ == "__main__":
    main()
