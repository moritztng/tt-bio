"""What `Trunk.launch`/`collect` spend outside the card: the 3.03 s a design `host_split` reads as
`prep_s` at 620 tokens (50 ms a sampler step), split into pad, mask upload, the s upload, the
conditioning uploads and the s readback.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/wh/prep_cost.py [--n 620] [--iters 20]

Shapes come from the released RFD4-Proteina config, not from weights: s_init [b, n, 768], one
conditioning tensor per recycle at [b, n, cond], the [b, n] mask. The trunk itself never runs, so
this is host and PCIe time alone and is honest on any card; the device only has to accept the
tensors. Numbers are medians over `--iters` rounds with a device sync around each timed part.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=620)
    ap.add_argument("--token-dim", type=int, default=768)
    ap.add_argument("--cond-dim", type=int, default=768)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    import ttnn

    from tt_bio import token_axis
    from tt_bio.rfd4.trunk import Trunk
    from tt_bio.tenstorrent import download, upload

    dev = ttnn.open_device(device_id=0)
    try:
        trunk = Trunk.__new__(Trunk)  # only the dtypes and the mask cache, never a pass
        trunk.dtype = ttnn.bfloat16
        trunk.residual_dtype = ttnn.bfloat16
        n = a.n
        npad = token_axis.pad_amount(n, token_axis.bucket_multiple("rfd4"))
        mask = torch.ones(1, n)
        s_init = torch.randn(1, n, a.token_dim)
        conds = [torch.randn(1, n, a.cond_dim) for _ in range(a.depth)]

        def sync():
            ttnn.synchronize_device(dev)

        def timed(fn):
            sync()
            t = time.perf_counter()
            r = fn()
            sync()
            return time.perf_counter() - t, r

        def pad(t, dims):
            for d in dims if npad else ():
                t = torch.nn.functional.pad(t, [0, 0] * (t.dim() - 1 - d) + [0, npad])
            return t

        def up(t, dtype):
            return upload(t.to(torch.float32).contiguous(), dtype=dtype, device=dev)

        got = {k: [] for k in ("pad_s", "masks_s", "up_s_s", "up_conds_s", "down_s_s")}
        for _ in range(a.iters):
            dt, m = timed(lambda: pad(mask.to(torch.float32), [1]))
            got["pad_s"].append(dt)
            dt, masks = timed(lambda: trunk.masks_for(m, dev))
            got["masks_s"].append(dt)
            mc = m[..., None]
            dt, s = timed(lambda: up(pad(s_init.float(), [1]) * mc, trunk.residual_dtype))
            got["up_s_s"].append(dt)
            dt, cs = timed(lambda: [up(pad(c.float(), [1]), trunk.dtype) for c in conds])
            got["up_conds_s"].append(dt)
            dt, _ = timed(lambda: download(s)[:, :n])
            got["down_s_s"].append(dt)
            for t in (s, *cs):
                if t.is_allocated():
                    ttnn.deallocate(t)

        ms = {k: round(statistics.median(v) * 1e3, 3) for k, v in got.items()}
        rec = {"n": n, "npad": npad, "token_dim": a.token_dim, "cond_dim": a.cond_dim,
               "depth": a.depth, "iters": a.iters, "ms_per_step": ms,
               "step_total_ms": round(sum(ms.values()), 3),
               "s_per_design_60_steps": round(sum(ms.values()) * 60 / 1e3, 3)}
        print(json.dumps(rec, indent=2))
        if a.out:
            Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
