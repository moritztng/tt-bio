#!/usr/bin/env python3
"""Does a batched-matmul program config ever return a wrong result? Repeat each (shape, config)
N times on fixed inputs and count calls whose output is non-finite or differs from the first."""
import argparse, json, sys, pathlib, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=60)
    ap.add_argument("--B", default="128,256")
    args = ap.parse_args()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio.autograd import precise_config
    dev = tt.get_device()
    grid = dev.compute_with_storage_grid_size()
    g = torch.Generator().manual_seed(1)
    nol1 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                            math_approx_mode=False, fp32_dest_acc_en=True,
                                            packer_l1_acc=False)
    res = {}

    def C(bw, sh, sw, pm, pn):
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid, in0_block_w=bw, out_subblock_h=sh,
            out_subblock_w=sw, per_core_M=pm, per_core_N=pn)

    for B in [int(x) for x in args.B.split(",")]:
        nq, nk, d, H = 128 if B == 128 else 256, 256, 32, 4
        T = lambda *s: ttnn.from_torch(torch.randn(*s, generator=g).to(torch.bfloat16),
                                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        q, k, P, v = T(B, H, nq, d), T(B, H, nk, d), T(B, H, nq, nk), T(B, H, nk, d)
        Mq = nq // 32
        cases = {
            "QKt": (q, k, False, True, {
                "cur(1,1,4,M,8)": (C(1, 1, 4, Mq, 8), precise_config()),
                "cur_nol1acc": (C(1, 1, 4, Mq, 8), nol1),
                "sub(1,1,2,M,8)": (C(1, 1, 2, Mq, 8), precise_config()),
                "pn4(1,1,4,M,4)": (C(1, 1, 4, Mq, 4), precise_config()),
                "pm2(1,1,4,2,8)": (C(1, 1, 4, 2, 8), precise_config()),
                "none": (None, precise_config()),
            }),
            "PV": (P, v, False, False, {
                "cur": (C(8, 4, 1, Mq, 1), precise_config()),
                "cur_nol1acc": (C(8, 4, 1, Mq, 1), nol1),
            }),
            "PtdO": (P, q, True, False, {
                "cur": (C(min(8, nq // 32), 4, 1, 8, 1), precise_config()),
            }),
        }
        for name, (a, b, ta, tb, variants) in cases.items():
            for vn, (pc, ck) in variants.items():
                bad_nf = bad_diff = 0
                first = None
                err = None
                t0 = time.time()
                try:
                    for i in range(args.N):
                        y = ttnn.to_torch(ttnn.matmul(a, b, transpose_a=ta, transpose_b=tb,
                                                      compute_kernel_config=ck, program_config=pc))
                        if not bool(torch.isfinite(y.float()).all()):
                            bad_nf += 1
                        elif first is None:
                            first = y
                        elif not torch.equal(y, first):
                            bad_diff += 1
                except Exception as e:
                    err = str(e)[:200]
                key = f"B{B}:{name}:{vn}"
                res[key] = dict(N=args.N, nonfinite=bad_nf, differs=bad_diff, error=err,
                                secs=time.time() - t0)
                print(f"{key:28s} nonfinite {bad_nf:3d}/{args.N} differs {bad_diff:3d} {err or ''}",
                      flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
