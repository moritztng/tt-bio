"""Other models digests for the two shared kernels the tri224 levers touch, run once per tree and compared.

T2 (`trimul_tail_res/reader_ln.cpp` now sizes the weight and gamma / beta pages by their own CB) at the keys other
models serve: c_z 128 plain (Boltz-2 / OpenFold3), 256 GATE (ESMFold2), 384 plain under `trimul_t2w` (OpenDDE). And
`permute_back_pk` (compute arg X8 / O8 = 0) in bf16, batch 1. Seeded bf16 inputs; sha256 of each output.
  python3 perf/rfd4/tri224/digest.py --n 256 1024 --out d.json
"""
import argparse
import hashlib
import json

import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
from tt_bio import mm_generic as MG  # noqa: E402
from tt_bio import permute_back_pk as PK  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import trimul_tail as TTL  # noqa: E402
from tt_bio.af2 import compute_kernel_config  # noqa: E402


def _dev(t, dev):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _sha(t):
    return hashlib.sha256(ttnn.to_torch(t).contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def t2(dev, n, c, gate, levers):
    gen = torch.Generator().manual_seed(n * 1000 + c + gate)
    h = {"x": torch.randn(1, n, n, c, generator=gen) * 2, "g": torch.randn(1, n, n, c, generator=gen),
         "z": torch.randn(1, n, n, c, generator=gen), "w": torch.randn(c, c, generator=gen) / c ** 0.5,
         "wg": torch.randn(c, c, generator=gen) / c ** 0.5, "gamma": 1 + 0.1 * torch.randn(c, generator=gen),
         "beta": 0.1 * torch.randn(c, generator=gen)}
    d = {k: _dev(v.to(torch.bfloat16), dev) for k, v in h.items()}
    kcfg = T.trunk_compute_kernel_config(compute_kernel_config())
    with T.lever_scope(levers):
        y = TTL.fused_tail_ln(d["x"], d["gamma"], d["beta"], d["w"], d["g"], d["z"], MG.ckc_args(kcfg),
                              tuple(T.COMPUTE_GRID_MAIN), wg=d["wg"] if gate else None)
    if y is None:
        return "declined " + str(dict(TTL.REJECTS))
    s = _sha(y)
    for t in d.values():
        ttnn.deallocate(t)
    return s


def pk(dev, n, c):
    x = _dev(torch.randn(1, c, n, n, generator=torch.Generator().manual_seed(n + c)).to(torch.bfloat16), dev)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, n, c]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                         ttnn.DRAM_MEMORY_CONFIG)
    s = _sha(PK.permute_back_pk(x, out, tuple(T.COMPUTE_GRID_MAIN)))
    ttnn.deallocate(x)
    ttnn.deallocate(out)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    TTL.T2 = True
    dev = T.get_device()
    res = {}
    for n in a.n:
        res[f"t2_c128_{n}"] = t2(dev, n, 128, False, ())
        res[f"t2_c256_gate_{n}"] = t2(dev, n, 256, True, ())
        res[f"t2_c384_t2w_{n}"] = t2(dev, n, 384, False, ("trimul_t2w",))
        res[f"pk_c128_{n}"] = pk(dev, n, 128)
        print(n, {k: v for k, v in res.items() if k.endswith(str(n))}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
