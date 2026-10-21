"""`dit_cond_sigat` on Boltz-2's token DiT (512 x 1): the adaLN scale/shift as one [768, 3072] linear without a
sigmoid epilogue, the sigmoid taken inside addcmul_at, against the shipped dit_cond_jcat path. Device time and
error against a float64 reference.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_gap_small/sigat.py --out sigat.jsonl

  jcat_lin    the shipped layer's three conditioning linears ([768, 1536] each; scale and gate with their sigmoid)
  sigat_lin   the lever's two ([768, 3072] scale|shift, [768, 1536] gate, no activation)
  jcat_mod    `sh + a * sc` of the first half through addcmul_at on the post-sigmoid scale
  sigat_mod   the same with sig_z: `sh + a * sigmoid(logit)` in the kernel
  *_e2e       linears + both halves' modulation, the error of the whole chain (rel only, time is the sum above)
rel = ||out - ref|| / ||ref||, ref in float64 from the same bf16 operands. Per dtype of the activation.
"""
import argparse, json, time, traceback

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from perf.spd_difflin.op_probe import dev_time, opened_nodes  # noqa: E402

N, K, D = 512, 768, 768


def rel(t, ref):
    x = ttnn.to_torch(t).double().reshape(ref.shape)
    return float((x - ref).norm() / ref.norm())


def bf(t):
    return t.bfloat16().double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = T.get_device()
    nodes = opened_nodes()
    torch.manual_seed(0)
    s_h, a_h = bf(torch.randn(N, K)), bf(torch.randn(N, D) * 1.5)
    sw, bw, ow = (bf(torch.randn(K, 2 * D) * 0.03) for _ in range(3))
    sb, ob = bf(torch.randn(2 * D) * 0.5), bf(torch.randn(2 * D) * 0.5)
    sc_ref, sh_ref = torch.sigmoid(s_h @ sw + sb), s_h @ bw
    go_ref = torch.sigmoid(s_h @ ow + ob)
    mod_ref = sh_ref[:, :D] + a_h * sc_ref[:, :D]
    e2e_ref = torch.cat((mod_ref, sh_ref[:, D:] + a_h * sc_ref[:, D:], a_h + a_h * go_ref[:, :D]), -1)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    rows = []
    for dt in (ttnn.bfloat16, ttnn.float32):
        up = lambda t, dt=dt: ttnn.from_torch(t.float().reshape(1, *t.shape) if t.dim() == 2 else t.float(),  # noqa: E731
                                              layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
        upw = lambda t: ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)  # noqa: E731
        s, a = up(s_h), up(a_h)
        w_sc, b_sc, w_sh, w_o, b_o = upw(sw), upw(sb), upw(bw), upw(ow), upw(ob)
        w_ss, b_ss = upw(torch.cat((sw, bw), -1)), upw(torch.cat((sb, torch.zeros_like(sb)), -1))
        lin = lambda x, w, b=None, act=None: ttnn.linear(x, w, bias=b, compute_kernel_config=ckc, activation=act)  # noqa: E731
        sc, sh, ss = lin(s, w_sc, b_sc, "sigmoid"), lin(s, w_sh), lin(s, w_ss, b_ss)

        def jcat_lin():
            outs = [lin(s, w_sc, b_sc, "sigmoid"), lin(s, w_sh), lin(s, w_o, b_o, "sigmoid")]
            for o in outs[1:]:
                ttnn.deallocate(o)
            return outs[0]

        def sigat_lin():
            outs = [lin(s, w_ss, b_ss), lin(s, w_o, b_o)]
            ttnn.deallocate(outs[1])
            return outs[0]

        def jcat_mod():
            return T._addcmul_at(sh, a, sc, D)

        def sigat_mod():
            return T._addcmul_at(ss, a, ss, D, x_off=2 * D, sig_z=True)

        def e2e(sig):
            if sig:
                l, o = lin(s, w_ss, b_ss), lin(s, w_o, b_o)
                parts = [T._addcmul_at(l, a, l, D, x_off=2 * D, sig_z=True),
                         T._addcmul_at(l, a, l, D, x_off=3 * D, z_off=D, sig_z=True),
                         T._addcmul_at(a, a, o, D, sig_z=True)]
                tmp = [l, o]
            else:
                c, h, o = lin(s, w_sc, b_sc, "sigmoid"), lin(s, w_sh), lin(s, w_o, b_o, "sigmoid")
                parts = [T._addcmul_at(h, a, c, D), T._addcmul_at(h, a, c, D, x_off=D, z_off=D),
                         T._addcmul_at(a, a, o, D)]
                tmp = [c, h, o]
            for t in tmp:
                ttnn.deallocate(t)
            out = ttnn.concat(parts, -1)
            for t in parts:
                ttnn.deallocate(t)
            return out

        arms = {"jcat_lin": (jcat_lin, None), "sigat_lin": (sigat_lin, None),
                "jcat_mod": (jcat_mod, mod_ref), "sigat_mod": (sigat_mod, mod_ref),
                "jcat_e2e": (lambda: e2e(False), e2e_ref), "sigat_e2e": (lambda: e2e(True), e2e_ref)}
        for arm, (fn, ref) in arms.items():
            r = dict(arm=arm, dtype=str(dt), arch=T.arch_name(), t=time.time())
            try:
                y = fn()
                if ref is not None:
                    r["rel"] = rel(y, ref)
                ttnn.deallocate(y)
                if not arm.endswith("_e2e"):
                    us, spread, clk = dev_time(dev, fn, nodes=nodes)
                    r.update(us=us, spread=spread, aiclk=clk)
            except Exception as e:  # noqa: BLE001 -- an arm that cannot run is a result, not a crash
                r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
                traceback.print_exc()
            rows.append(r)
            print(json.dumps(r), flush=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
