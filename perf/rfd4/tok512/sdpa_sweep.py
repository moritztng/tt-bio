"""q/k chunk sweep of the RFD4 trunk's pair-biased SDPA under the device profiler, with a bit-identity check per rung.

    SCRIPT=sdpa_sweep.py perf/rfd4/tok512/mm_sweep.sh <out dir> <card> [--shapes 1x512 10x224 4x320 8x256 32x128]

Shapes are batch x tokens, 12 heads of 64, bias [b, 12, n, n], the trunk's kernel config. A rung whose q chunk
alone differs from the shipped one must give torch.equal outputs (q chunks split independent rows); a k chunk
change moves the online-softmax order and is reported with its float64 rel RMS.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402
from tracy import signpost  # noqa: E402

import clocksample  # noqa: E402

H, D = 12, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", nargs="+", default=["1x512", "10x224", "4x320", "8x256", "32x128"])
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()

    from tt_bio.rfd4.trunk import compute_kernel_config
    from tt_bio.tenstorrent import (_padded_sdpa_len, _sdpa_program_config, _sdpa_program_config_for_lengths,
                                    fused_sdpa, get_device)

    dev = get_device()
    ckc = compute_kernel_config()
    g = torch.Generator().manual_seed(0)
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    res = []
    signpost("elt_prof")
    with clocksample.during() as clk:
        for sh in a.shapes:
            b, n = map(int, sh.split("x"))
            q, k, v = (torch.randn(b, H, n, D, generator=g) for _ in range(3))
            bias = torch.randn(b, H, n, n, generator=g)
            qd, kd, vd = (x.bfloat16().double() for x in (q, k, v))
            ref = torch.softmax(qd @ kd.transpose(-1, -2) / D ** 0.5 + bias.bfloat16().double(), -1) @ vd
            qt, kt, vt, bt = map(up, (q, k, v, bias))
            shipped = _sdpa_program_config_for_lengths(n, n, site="tok512")
            rule = _sdpa_program_config_for_lengths(n, n, b * H, site="tok512")
            p = _padded_sdpa_len(n)
            rungs = {(shipped.q_chunk_size, shipped.k_chunk_size), (rule.q_chunk_size, rule.k_chunk_size)}
            rungs |= {(qc, shipped.k_chunk_size) for qc in range(32, min(p, 512) + 1, 32) if p % qc == 0}
            base = None
            for qc, kc in sorted(rungs, key=lambda r: (r != (shipped.q_chunk_size, shipped.k_chunk_size), r)):
                cfg = _sdpa_program_config(q_chunk_size=qc, k_chunk_size=kc)
                fn = lambda: fused_sdpa(qt, kt, vt, attn_mask=bt, scale=D ** -0.5,  # noqa: E731
                                        compute_kernel_config=ckc, program_config=cfg)
                label = f"{b}x{n}|q{qc}k{kc}"
                try:
                    y = ttnn.to_torch(fn())
                except RuntimeError as e:
                    res.append({"label": label, "refused": str(e).splitlines()[0][:160]})
                    continue
                if base is None:
                    base = y
                err = float(((y.double() - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
                ttnn.synchronize_device(dev)
                signpost(f"B|{label}")
                for _ in range(a.reps):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(dev)
                signpost(f"E|{label}")
                ttnn.ReadDeviceProfiler(dev)
                r = {"label": label, "err": err, "equal_shipped": bool(torch.equal(y, base)),
                     "shipped": [shipped.q_chunk_size, shipped.k_chunk_size],
                     "rule": [rule.q_chunk_size, rule.k_chunk_size]}
                res.append(r)
                print(json.dumps(r), flush=True)
            for t in (qt, kt, vt, bt):
                ttnn.deallocate(t)
    signpost("elt_end")
    Path(a.out).write_text(json.dumps(dict(rows=res, reps=a.reps, aiclk=clk.summary(),
                                           load=clk.load and max(clk.load)), indent=1))
    print(clk.line())


if __name__ == "__main__":
    main()
