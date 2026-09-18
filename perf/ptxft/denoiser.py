#!/usr/bin/env python3
"""The diffusion module at ONE sampled timestep, differentiated end to end.

This is the assembly. Both block types it composes are already checked against
ByteDance's own module in float64 -- the 24 token blocks at <= 7.91e-03 and the 6 atom
blocks at <= 1.56e-02 (perf/ptxft/ditcheck.py) -- so what is unverified here is the
WIRING, and the wiring is what this harness checks, against the production forward it
mirrors line for line (DiffusionModule._denoise_device, protenix.py:1092-1130).

ONE TIMESTEP, which is what diffusion training differentiates. sigma is drawn from
Protenix's own TrainingNoiseSampler (generator.py:49-60, exp(N(-1.2, 1.5)) * 16) and the
denoiser is called once. The standing NO-GO is on backpropagating the 200-step INFERENCE
rollout, whose mechanism is chained bf16 numerics over 200 denoise calls; there is no
trajectory in this graph at all.

TWO CHECKS, because a forward match alone would not test the backward and an analytic
gradient alone would not test the assembly:

  FORWARD   the taped twin against production's own _denoise_device on the same
            conditioning, the same r_noisy and the same sigma.
  GRADIENT  a directional finite difference taken through PRODUCTION's forward -- two
            production calls at r +/- eps*d -- against the twin's analytic gradient
            dotted with the same d. Nothing on the twin side enters the reference.

The conditioning (ss_base, c_la, p, Smean, S, dit_z) is captured from a live fold, which
is where it comes from in training too: it is a function of the trunk, and the trunk's
own gradients are checked separately. It enters the tape as non-gradient leaves.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

SIGMA_DATA = 16.0


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / d) if d > 0 else float(np.linalg.norm(a - b))


def cosine(a, b):
    a, b = np.ravel(np.asarray(a, np.float64)), np.ravel(np.asarray(b, np.float64))
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else 1.0


class TapedDenoiser:
    """_denoise_device over tt_bio.autograd, mirroring protenix.py:1092-1130 in order."""

    def __init__(self, sd, device, *, dtype, cond, n_dit=24):
        import ttnn
        from perf.ptxft.tape_block import DiTBlock
        from tt_bio import autograd as ag
        from tt_bio import finetune as ft
        self.ag, self.ft, self.ttnn = ag, ft, ttnn
        self.device, self.dtype = device, dtype
        self.w = {}

        def take(key, arr=None):
            a = sd[key] if arr is None else arr
            a = a.detach().to("cpu").float().numpy()
            a = a.T if a.ndim == 2 else a.reshape(1, 1, -1)
            self.w[key] = ag.Tensor(ft.to_device(np.ascontiguousarray(a), device,
                                                 dtype=dtype), requires_grad=False)

        for k in ("diffusion_conditioning.layernorm_n.weight",
                  "diffusion_conditioning.linear_no_bias_n.weight",
                  "atom_attention_encoder.linear_no_bias_r.weight",
                  "atom_attention_encoder.linear_no_bias_q.weight",
                  "layernorm_s.weight", "linear_no_bias_s.weight", "layernorm_a.weight",
                  "atom_attention_decoder.linear_no_bias_a.weight",
                  "atom_attention_decoder.layernorm_q.weight",
                  "atom_attention_decoder.linear_no_bias_out.weight"):
            take(k)
        for nm in ("transition_s1", "transition_s2"):
            P = f"diffusion_conditioning.{nm}."
            for k in ("layernorm1.weight", "layernorm1.bias", "linear_no_bias_a.weight",
                      "linear_no_bias_b.weight", "linear_no_bias.weight"):
                take(P + k)

        def blocks(prefix, n, **kw):
            out = []
            for i in range(n):
                pre = f"{prefix}{i}."
                bsd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
                out.append(DiTBlock(bsd, device, dtype=dtype, param_dtype=dtype,
                                    trainable=False, **kw))
            return out
        NQ, NK = 32, 128
        self.enc = blocks("atom_attention_encoder.atom_transformer."
                          "diffusion_transformer.blocks.", 3, n_queries=NQ, n_keys=NK)
        self.dec = blocks("atom_attention_decoder.atom_transformer."
                          "diffusion_transformer.blocks.", 3, n_queries=NQ, n_keys=NK)
        self.dit = blocks("diffusion_transformer.blocks.", n_dit)
        self.cond = cond

    # ------------------------------------------------------------------ pieces

    def _lin(self, x, k):
        return self.ag.linear(x, self.w[k])

    def _ln(self, x, k, b=None):
        return self.ag.layer_norm(x, self.w[k], None if b is None else self.w[b],
                                  eps=1e-5)

    def _transition(self, x, nm):
        ag = self.ag
        P = f"diffusion_conditioning.{nm}."
        xn = self._ln(x, P + "layernorm1.weight", P + "layernorm1.bias")
        a = self._lin(xn, P + "linear_no_bias_a.weight")
        b = self._lin(xn, P + "linear_no_bias_b.weight")
        return self._lin(ag.mul(ag.mul(a, ag.sigmoid(a)), b), P + "linear_no_bias.weight")

    def __call__(self, r_noisy, fou):
        ag, c = self.ag, self.cond
        nn_ = self._lin(self._ln(fou, "diffusion_conditioning.layernorm_n.weight"),
                        "diffusion_conditioning.linear_no_bias_n.weight")
        ss = ag.add(c["ss_base"], nn_)
        for nm in ("transition_s1", "transition_s2"):
            ss = ag.add(ss, self._transition(ss, nm))
        s_single = ss
        q_l = ag.add(c["c_la"],
                     self._lin(r_noisy, "atom_attention_encoder.linear_no_bias_r.weight"))
        x = q_l
        for b in self.enc:
            x = b(x, c["c_la"], c["p"])
        q_skip = x
        a_tok = ag.matmul(c["Smean"], ag.relu(
            self._lin(x, "atom_attention_encoder.linear_no_bias_q.weight")))
        a_tok = ag.add(a_tok, self._lin(self._ln(s_single, "layernorm_s.weight"),
                                        "linear_no_bias_s.weight"))
        a_t = a_tok
        for b in self.dit:
            a_t = b(a_t, s_single, c["dit_z"])
        a_t = self._ln(a_t, "layernorm_a.weight")
        q = ag.add(ag.matmul(c["S"], self._lin(
            a_t, "atom_attention_decoder.linear_no_bias_a.weight")), q_skip)
        for b in self.dec:
            q = b(q, c["c_la"], c["p"])
        qn = self._ln(q, "atom_attention_decoder.layernorm_q.weight")
        return self._lin(qn, "atom_attention_decoder.linear_no_bias_out.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.boltz/protenix-v2.pt"))
    ap.add_argument("--seq", default="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eps", type=float, default=2e-2)
    ap.add_argument("--fwd-bar", type=float, default=5.0e-2)
    ap.add_argument("--fd-bar", type=float, default=0.10)
    ap.add_argument("--n-dit", type=int, default=24)
    a = ap.parse_args()

    import torch
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio import autograd as ag
    from tt_bio import finetune as ft
    from tt_bio.protenix import Protenix
    from tt_bio.protenix_data import build_protein_features
    from perf.clocksample import during

    rng = np.random.default_rng(a.seed)
    fails = []
    with during() as clk:
        dev = get_device()
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        model = Protenix.load_from_checkpoint(a.ckpt, compute_kernel_config=ckc,
                                             device=dev)
        dm = model.diffusion if hasattr(model, "diffusion") else model.diff
        # Capture the conditioning from a live fold: _atom_cond is where the device
        # tensors the denoiser reads are built (protenix.py:965).
        grabbed = {}
        orig = type(dm)._atom_cond

        def spy(self, cond):
            out = orig(self, cond)
            grabbed.setdefault("cond", cond)
            grabbed.setdefault("dm", self)
            return out
        type(dm)._atom_cond = spy
        try:
            model.fold(build_protein_features(a.seq), n_step=1, n_sample=1, seed=a.seed)
        finally:
            type(dm)._atom_cond = orig
        cond = grabbed["cond"]
        dm = grabbed["dm"]
        N = int(cond["c_l"].shape[0])
        NT = int(cond["s_inputs"].shape[0])
        print(f"# --denoiser: {len(a.seq)} aa -> {N} atoms, {NT} tokens, "
              f"{a.n_dit} token blocks + 3 + 3 atom blocks")

        # One sampled sigma, upstream's own sampler.
        sigma = float(np.exp(rng.standard_normal() * 1.5 - 1.2) * SIGMA_DATA)
        t_hat = torch.tensor([sigma], dtype=torch.float32)
        wf = dm._w["diffusion_conditioning.fourier_embedding.w"]
        bf = dm._w["diffusion_conditioning.fourier_embedding.b"]
        tp = torch.log(t_hat / SIGMA_DATA) / 4
        fou = torch.cos(2 * math.pi * (tp.unsqueeze(-1) * wf + bf)).contiguous()
        r0 = (rng.standard_normal((N, 3)) * 1.0).astype(np.float32)
        print(f"# sigma {sigma:.4f} from TrainingNoiseSampler, eps {a.eps}")

        def production(r):
            out = dm._denoise_device(dm._up(torch.from_numpy(r)), dm._up(fou), cond)
            return np.asarray(ttnn.to_torch(out).float()).reshape(-1, 3)[:N]

        ref = production(r0)

        # The twin, on the same conditioning as non-gradient leaves.
        sd = {k[len("diffusion_module."):]: v for k, v in
              (lambda d: {(k[len("module."):] if k.startswith("module.") else k): v
                          for k, v in d.items()})(
                  (lambda s: next(s[k] for k in ("model", "state_dict", "ema", "module")
                                  if isinstance(s, dict) and k in s
                                  and isinstance(s[k], dict)) if any(
                      isinstance(s, dict) and k in s and isinstance(s[k], dict)
                      for k in ("model", "state_dict", "ema", "module")) else s)(
                      torch.load(a.ckpt, map_location="cpu", weights_only=False))
              ).items() if k.startswith("diffusion_module.")}
        # Rank matters and production fixes it at the call site: _denoise_device reshapes
        # c_la and the single stream to (1, N, c) before handing them to the atom
        # transformer. Feeding a rank-2 leaf into a block whose output is rank 3 makes
        # the residual broadcast, and the gradient then comes back one rank too wide.
        leaf = lambda t: ag.Tensor(t, requires_grad=False)
        r3 = lambda t, n: ag.Tensor(ttnn.reshape(t, (1, n, int(t.shape[-1]))),
                                    requires_grad=False)
        tcond = {"ss_base": r3(cond["ss_base"], NT), "c_la": r3(cond["c_la_dev"], N),
                 "p": leaf(cond["p_dev"]), "Smean": leaf(cond["Smean_dev"]),
                 "S": leaf(cond["S_dev"]), "dit_z": leaf(cond["dit_z"])}
        twin = TapedDenoiser(sd, dev, dtype=ttnn.bfloat16, cond=tcond, n_dit=a.n_dit)
        rt = ag.Tensor(ft.to_device(r0.reshape(1, N, 3), dev, dtype=ttnn.bfloat16),
                       requires_grad=True)
        fout = leaf(ft.to_device(np.asarray(fou), dev, dtype=ttnn.bfloat16))
        out = twin(rt, fout)
        got = ft.to_host(out.value).reshape(-1, 3)[:N]
        print()
        print(f"{'quantity':<40} {'value':>12} {'bar':>10}")
        fr = rel_l2(got, ref)
        fc = cosine(got, ref)
        print(f"{'forward vs production, rel L2':<40} {fr:>12.3e} {a.fwd_bar:>10.1e}"
              f"{'' if fr <= a.fwd_bar else '   <-- FAIL'}")
        print(f"{'forward vs production, cos':<40} {fc:>12.6f}")
        if not (fr <= a.fwd_bar):
            fails.append(f"forward rel L2 {fr:.3e}")

        # Directional finite difference THROUGH PRODUCTION.
        g = (rng.standard_normal((N, 3)) * 0.1).astype(np.float32)
        d = rng.standard_normal((N, 3)).astype(np.float32)
        d /= np.linalg.norm(d)
        full_g = np.zeros_like(r0)
        full_g[:N] = g
        out.backward(seed=ft.to_device(full_g.reshape(1, N, 3), dev,
                                       dtype=ttnn.bfloat16))
        an = float((np.float64(ft.to_host(rt.grad).reshape(-1, 3)[:N]) *
                    np.float64(d)).sum())
        lp = float((np.float64(production(r0 + a.eps * d)) * np.float64(g)).sum())
        lm = float((np.float64(production(r0 - a.eps * d)) * np.float64(g)).sum())
        fd = (lp - lm) / (2 * a.eps)
        # A NaN must FAIL. `nan > bar` is False, so a comparison alone would have
        # reported PASS on a gradient that does not exist -- it did, once.
        err = float("inf") if not np.isfinite(an) else abs(an - fd) / max(abs(fd), 1e-30)
        print(f"{'directional FD through production':<40} {fd:>12.6f}")
        print(f"{'twin analytic gradient, same direction':<40} {an:>12.6f}")
        print(f"{'relative disagreement':<40} {err:>12.3e} {a.fd_bar:>10.1e}"
              f"{'' if err <= a.fd_bar else '   <-- FAIL'}")
        if not (err <= a.fd_bar):
            fails.append(f"directional FD disagrees by {err:.3e}"
                         + ("" if np.isfinite(an) else "; the analytic gradient is NaN"))
    print()
    print(clk.line(0))
    print()
    if fails:
        print(f"DENOISER FAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("DENOISER PASS: the assembled denoiser matches production forward and its "
          "gradient matches a finite difference taken through production")
    return 0


if __name__ == "__main__":
    sys.exit(main())
