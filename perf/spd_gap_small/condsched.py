"""Device time per diffusion step of Boltz-2's token DiT conditioning projections, three ways.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_gap_small/condsched.py --out condsched.jsonl

dit_sched (batching the conditioning of 8 steps) read null on the WH --fast fold (dsab, +0.08 s median). The
arms here say why: `jcat` is what WH --fast ships (24 layers x three [512, 768] x [768, 1536] linears a step),
`shoist` the three wide [512, 768] x [768, 36864] linears a step, `sched` the same wide linears over 8 steps'
rows at once, divided by 8. Same compute kernel config as the model, at HiFi4 and HiFi2. Values do not matter,
shapes do. Device time is a captured trace replayed back to back (op_probe.dev_time).
"""
import argparse, json, time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from perf.spd_difflin.op_probe import dev_time, opened_nodes  # noqa: E402

N, D, L, K = 512, 768, 24, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = T.get_device()
    nodes = opened_nodes()
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)  # noqa: E731
    torch.manual_seed(0)
    s1, s8 = up(torch.randn(1, N, D)), up(torch.randn(K, N, D))
    wj = [(up(torch.randn(D, 2 * D) * .03), up(torch.randn(1, 2 * D))) for _ in range(L)]
    ww = [(up(torch.randn(D, 2 * L * D) * .03), up(torch.randn(1, 2 * L * D))) for _ in range(3)]
    rows = []
    for fid in ("HiFi4", "HiFi2"):
        ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                               fp32_dest_acc_en=True, packer_l1_acc=True)
        sig = T.mm_sigmoid(ckc)

        def jcat():
            outs = []
            for w, b in wj:
                outs += [ttnn.linear(s1, w, bias=b, compute_kernel_config=ckc, activation=sig),
                         ttnn.linear(s1, w, compute_kernel_config=ckc),
                         ttnn.linear(s1, w, bias=b, compute_kernel_config=ckc, activation=sig)]
            for o in outs[:-1]:
                ttnn.deallocate(o)
            return outs[-1]

        def wide(s):
            def f():
                o = [ttnn.linear(s, w, bias=b if i != 1 else None, compute_kernel_config=ckc, activation=sig if i != 1 else None)
                     for i, (w, b) in enumerate(ww)]
                ttnn.deallocate(o[0]); ttnn.deallocate(o[1])
                return o[2]
            return f

        for arm, fn, per in (("jcat", jcat, 1), ("shoist", wide(s1), 1), ("sched", wide(s8), K)):
            us, spread, clk = dev_time(dev, fn, nodes=nodes)
            r = dict(arm=arm, fidelity=fid, us_per_step=us / per, spread=spread, aiclk=clk, arch=T.arch_name(), t=time.time())
            rows.append(r)
            print(json.dumps(r), flush=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
