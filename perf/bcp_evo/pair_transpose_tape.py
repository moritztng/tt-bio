#!/usr/bin/env python3
"""`pair_transpose` under the tape: y = transpose(x) * w through `tenstorrent._pair_transpose`,
seeded with a fixed cotangent. Off = no entry (the ROW_MAJOR route the round runs), on = the
`pair_transpose` entry (kernel forward, kernel VJP). Pass bar: y and x.grad torch.equal arm to arm
and the kernel served forward and backward on the on arm only.
"""
import json, os, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import autograd as ag, pair_transpose as PT, taped_ttnn as T, tenstorrent as TN
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    D = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=dt,  # noqa: E731
                                                     device=dev, memory_config=D)
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).float()  # noqa: E731
    res = {}
    for shape in ((288, 288, 128), (256, 256, 64), (512, 512, 128)):
        torch.manual_seed(0)
        x = up(torch.randn(shape))
        tshape = list(shape)
        tshape[-3], tshape[-2] = tshape[-2], tshape[-3]
        w, seed = up(torch.randn(tshape)), up(torch.randn(tshape))
        arms = {}
        for on in (False, True):
            os.environ["TT_BIO_TAPED_KERNELS"] = "pair_transpose" if on else ""
            PT.PAIR_TRANSPOSE_FUSED = on
            s0 = list(PT.STATS)
            with T.tape():
                xt = ag.Tensor(x, requires_grad=True)
                y = T._SHIM.multiply(TN._pair_transpose(xt, D), w)
            yv = down(y.value)
            ag.backward([y], [seed])
            arms[on] = (yv, down(xt.grad), [a - b for a, b in zip(PT.STATS, s0)])
        rec = {"y_equal": bool(torch.equal(arms[False][0], arms[True][0])),
               "grad_equal": bool(torch.equal(arms[False][1], arms[True][1])),
               "stats_off": arms[False][2], "stats_on": arms[True][2]}
        res["x".join(map(str, shape))] = rec
        print(json.dumps({"x".join(map(str, shape)): rec}), flush=True)
    (ROOT / "perf/bcp_evo/out/pair_transpose_tape.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
