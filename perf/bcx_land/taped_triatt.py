#!/usr/bin/env python3
"""OpenFold3's trunk TriangleAttention under a tape, then untaped, in one process.

The construction is openfold3_trunk.py's: fp32_softmax, scale_pair_bias=False, no linear biases,
fused_hifi on. That one module reaches all three sites wk/bcx-land changes: the fused-HiFi ladder
(`_tri_att_sdpa_hifi_inner`), the fp32-softmax block plan (`_fp32_softmax_attention`) and the
`triatt_qkv` generic_op entry points.

  1. untaped forward, y0, and the fused-HiFi served count
  2. taped forward + backward, dL/dz
  3. untaped forward again, y1: must equal y0 bit for bit and must still be served fused
     (the latch: on main a taped decline retired the config, so step 3 declined)
  4. dL/dz graded against finite differences through the same module's UNTAPED forward, eps
     swept, along the gradient's own direction (perf/ptx_fastpath/modulecheck.py's method), with
     a negative control that scales the analytic side 1.5x and must be rejected

AICLK is read from sysfs (`tt_aiclk`), not tt-smi, which hangs on qb1's dead card.
"""
import argparse, json, math, os, threading, time, traceback
import numpy as np
import torch

BAR = 2.0e-1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--sysfs", default="/sys/class/tenstorrent/tenstorrent!0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    clk, stop = [], threading.Event()
    def sample():
        while not stop.is_set():
            try:
                clk.append(int(open(os.path.join(a.sysfs, "tt_aiclk")).read().strip()))
            except Exception:
                pass
            time.sleep(0.5)
    threading.Thread(target=sample, daemon=True).start()

    rng = np.random.default_rng(0)
    S, c_z, heads, hd = a.tokens, 128, 4, 32
    t = lambda *sh: torch.tensor(rng.standard_normal(sh) / math.sqrt(sh[-1]), dtype=torch.float32)
    sd = {"layer_norm.weight": torch.tensor(rng.standard_normal(c_z) * 0.2 + 1.0, dtype=torch.float32),
          "layer_norm.bias": torch.tensor(rng.standard_normal(c_z) * 0.05, dtype=torch.float32),
          "linear_q.weight": t(heads * hd, c_z), "linear_k.weight": t(heads * hd, c_z),
          "linear_v.weight": t(heads * hd, c_z), "linear_g.weight": t(heads * hd, c_z),
          "linear_o.weight": t(c_z, heads * hd), "linear.weight": t(heads, c_z)}
    dev = tt.get_device()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    mod = tt.TriangleAttention(hd, heads, False, sd, ckc, scale_pair_bias=False,
                               fp32_softmax=True, fused_hifi=True)
    D = lambda x: ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
    x0 = torch.tensor(rng.standard_normal((1, S, S, c_z)), dtype=torch.float32)
    lw = None
    st = tt.TRIATT_FUSED_HIFI_STATS
    rep = {"tokens": S, "commit": os.popen("git rev-parse --short HEAD").read().strip()}

    y0 = ttnn.to_torch(mod(D(x0))).float()
    rep["served_untaped_1"] = st["served"]
    try:
        xa = ag.Tensor(D(x0), requires_grad=True)
        with ag.tape():
            out = mod(xa)
        lw = torch.tensor(rng.standard_normal([int(v) for v in out.value.shape]), dtype=torch.float32)
        out.backward(seed=D(lw))
        g = ttnn.to_torch(xa.grad).to(torch.float64)
        rep["taped"] = "ran"
        rep["grad_norm"] = float(g.norm())
    except Exception as e:
        rep["taped"] = f"raised {type(e).__name__}: {str(e)[:200]}"
        g = None
    rep["served_after_tape"] = st["served"]
    rep["over_l1_after_tape"] = len(tt._TRIATT_HIFI_OVER_L1)
    y1 = ttnn.to_torch(mod(D(x0))).float()
    rep["served_untaped_2"] = st["served"]
    rep["untaped_after_tape_served_fused"] = st["served"] > rep["served_after_tape"]
    rep["y1_equals_y0"] = bool(torch.equal(y0, y1))

    if g is not None:
        d = (g / g.abs().max()).to(torch.float32)
        analytic = float((g * d.to(torch.float64)).sum())
        def loss(eps):
            o = ttnn.to_torch(mod(D(x0 + eps * d))).to(torch.float64)
            return float((o * lw.to(torch.float64)).sum())
        rows = []
        for eps in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
            num = (loss(eps) - loss(-eps)) / (2 * eps)
            rows.append((eps, num, abs(num - analytic) / max(abs(analytic), 1e-12)))
        best = min(rows, key=lambda r: r[2])
        bad = 1.5 * analytic
        badrel = min(abs(n - bad) / abs(bad) for _, n, _ in rows)
        rep.update(analytic=analytic, fd=[list(r) for r in rows], fd_best_rel=best[2],
                   fd_best_eps=best[0], negctl_rel=badrel,
                   fd_pass=best[2] <= BAR and badrel > BAR)
    stop.set()
    rep["aiclk_during"] = [min(clk), max(clk), len(clk)] if clk else None
    rep["loadavg"] = open("/proc/loadavg").read().split()[:3]
    print(json.dumps(rep, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
