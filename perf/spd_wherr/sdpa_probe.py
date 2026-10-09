"""Wrong-pixel rate of the fp32 DiT attention (tt_bio.sdpa_generic, `_sdpa32`) per q/k chunk, vs float64.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_wherr/sdpa_probe.py --out probe.jsonl [--nt 730] [--draws 4]

Inside the kernel the score matmul accumulates head_dim / 32 tiles in dest and the PV matmul k_chunk / 32 tiles,
both under fp32 dest acc, which is the Wormhole erratum's exposure. A smaller k chunk shortens the PV run (32 is one
tile, the PV clean case); the score run is fixed by head_dim. Inputs as perf/spd_diffusion/sdpa32.py builds them
(token axis padded to a chunk multiple, padded keys masked at -1e4, bias / scale as the mask). A pixel is wrong under
perf/spd_wherr/audit.py's rule: error above 8 bf16 ulps of the float64 value and above 16x the call's rms error.
Per cell: wrong, wrong with error > 0.25, max error, rms error, and the device time per call (median of 5, warm).
"""
import argparse, json, statistics, time
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio import sdpa_generic as SG  # noqa: E402


def wrong(R, Y):
    err = (Y - R).abs()
    rms = err.pow(2).mean().sqrt().item()
    ulp = torch.exp2(torch.floor(torch.log2(R.abs().clamp_min(1e-30))) - 7)
    bad = (err > 8 * ulp) & (err > 16 * rms)
    worst = [[round(float(R[bad][i]), 4), round(float(Y[bad][i]), 4)]
             for i in err[bad].topk(min(3, int(bad.sum()))).indices.tolist()] if bad.any() else []
    return int(bad.sum()), int((bad & (err > 0.25)).sum()), float(err.max()), rms, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nt", type=int, default=730)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ls", type=float, default=2.0, help="q, k, v scale")
    ap.add_argument("--chunks", default="256x256,256x128,256x64,256x32,128x32")
    ap.add_argument("--draws", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(6)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size(); grid = (g.x, g.y)
    np_ = -(-a.nt // 256) * 256
    s = a.dim ** -0.5
    log = open(a.out, "a")

    def up(t):
        return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)

    for d in range(a.draws):
        gen = torch.Generator().manual_seed(d)
        q, k, v = (torch.randn(a.m, a.heads, a.nt, a.dim, generator=gen) * a.ls for _ in range(3))
        bias = torch.randn(1, a.heads, a.nt, a.nt, generator=gen) * 3
        R = torch.softmax(torch.einsum("mhid,mhjd->mhij", q.double(), k.double()) * s + bias.double(), -1) @ v.double()
        pad = lambda t: torch.nn.functional.pad(t, (0, 0, 0, np_ - a.nt))
        mask = torch.full((1, a.heads, np_, np_), -1e4); mask[:, :, :, :a.nt] = 0; mask[:, :, :a.nt, :a.nt] = bias / s
        tq, tk, tv, tm = up(pad(q)), up(pad(k)), up(pad(v)), up(mask)
        for c in a.chunks.split(","):
            qc, kc = map(int, c.split("x"))
            if np_ % qc or np_ % kc:
                continue

            def run():
                out = ttnn.allocate_tensor_on_device(ttnn.Shape([a.m, a.heads, np_, a.dim]), ttnn.float32,
                                                     ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
                return SG.sdpa(dev, tq, tk, tv, tm, out, qc, kc, grid, T._SDPA32_CKC, s)
            try:
                y = run()
            except Exception as e:
                cell = dict(draw=d, chunk=c, error=f"{type(e).__name__}: {str(e)[:200]}")
                log.write(json.dumps(cell) + "\n"); log.flush(); print(json.dumps(cell), flush=True); continue
            Y = torch.Tensor(ttnn.to_torch(y)).double()[:, :, :a.nt]; ttnn.deallocate(y)
            nb, nbig, mx, rms, worst = wrong(R, Y)
            us = None
            if d == 0:
                ts = []
                for _ in range(5):
                    ttnn.synchronize_device(dev); t0 = time.perf_counter(); y = run(); ttnn.synchronize_device(dev)
                    ts.append(time.perf_counter() - t0); ttnn.deallocate(y)
                us = round(statistics.median(ts) * 1e6, 1)
            cell = dict(draw=d, chunk=c, nt=a.nt, dim=a.dim, wrong=nb, big=nbig, max_err=round(mx, 5),
                        rms_err=round(rms, 7), worst=worst, us=us, arch=str(dev.arch()))
            log.write(json.dumps(cell) + "\n"); log.flush(); print(json.dumps(cell), flush=True)
        for t in (tq, tk, tv, tm):
            ttnn.deallocate(t)


if __name__ == "__main__":
    main()
