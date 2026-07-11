"""Benchmark current vs raw trimul channel moves on the real 48-block ESMFold2 trunk.

The benchmark uses checkpoint weights and identical deterministic pair inputs. Timed
regions contain only the device-resident trunk and an explicit device synchronize;
host/device transfers and output comparisons are outside the timing window.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch
import ttnn


def _compare(
    a: torch.Tensor, b: torch.Tensor, chunk: int = 1 << 22
) -> dict[str, float | bool]:
    x = a.reshape(-1)
    y = b.reshape(-1)
    n = x.numel()
    sx = sy = sxx = syy = sxy = 0.0
    max_abs = 0.0
    finite = True
    for start in range(0, n, chunk):
        xf = x[start : start + chunk].float()
        yf = y[start : start + chunk].float()
        finite = finite and bool(torch.isfinite(xf).all() and torch.isfinite(yf).all())
        d = (xf - yf).abs()
        max_abs = max(max_abs, float(d.max()))
        xd, yd = xf.double(), yf.double()
        sx += float(xd.sum())
        sy += float(yd.sum())
        sxx += float((xd * xd).sum())
        syy += float((yd * yd).sum())
        sxy += float((xd * yd).sum())
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    pcc = cov / max((vx * vy) ** 0.5, 1e-30)
    return {"pcc": pcc, "max_abs": max_abs, "finite": finite}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--checkpoint", default="biohub/ESMFold2")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(20260711)

    from tt_bio import esmfold2 as E
    from tt_bio import tenstorrent as T
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model

    print("loading checkpoint and 48-block folding trunk", flush=True)
    reference = ESMFold2Model.from_pretrained(args.checkpoint, load_esmc=False).eval()
    prefix = "folding_trunk."
    trunk_state = {
        key[len(prefix):]: value.float()
        for key, value in reference.state_dict().items()
        if key.startswith(prefix)
    }
    n_layers = reference.config.folding_trunk.n_layers
    del reference
    gc.collect()

    trunk = E.FoldingTrunk(n_layers=n_layers)
    trunk.load_state_dict(trunk_state, strict=False)
    del trunk_state
    gc.collect()
    assert n_layers == 48, f"expected production 48-block trunk, got {n_layers}"

    def execute(z: torch.Tensor, raw: bool) -> tuple[float, torch.Tensor]:
        T._TRIMUL_RAW_CHANNEL_MOVES = raw
        z_tt = trunk._from_torch(z)
        ttnn.synchronize_device(trunk.tt_device)
        started = time.perf_counter()
        out_tt = trunk.module(z_tt, None)
        ttnn.synchronize_device(trunk.tt_device)
        elapsed = time.perf_counter() - started
        out = trunk._to_torch(out_tt)
        ttnn.deallocate(out_tt)
        return elapsed, out

    for size in args.sizes:
        print(f"warming N={size} current and raw paths", flush=True)
        generator = torch.Generator().manual_seed(20260711 + size)
        z = torch.randn((1, size, size, E.C_Z), generator=generator)
        for raw in (False, True):
            _, warm = execute(z, raw)
            del warm
            gc.collect()

        current_s, current = execute(z, False)
        raw_s, raw = execute(z, True)
        metrics = _compare(current, raw)
        record = {
            "N": size,
            "blocks": n_layers,
            "current_s": current_s,
            "raw_s": raw_s,
            "speedup_current_over_raw": raw_s / current_s,
            **metrics,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        del z, current, raw
        gc.collect()


if __name__ == "__main__":
    main()
