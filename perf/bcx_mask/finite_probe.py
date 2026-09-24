"""Can the masked path produce a NON-FINITE value? Every site, forward and backward, measured.

`bcx-predictor`'s design loop leaves after about one round of fifty, and the candidate exit is
`trajectory.py:134-136`, `if not math.isfinite(design_loss): break`. A pLDDT that agrees with the
lab's to 1 % says nothing about one inf in a tensor whose summary is taken after a reduction.

There are exactly three places in an Evoformer block where a masked fold can put a small number
in a denominator, and this probe measures each rather than arguing about it:

  * the outer product mean's `eps + sum_s m_si m_sj` (`af2_reference.py:287`), which is `eps`
    alone on every pair that touches a pad residue -- pad-pad AND real-pad;
  * every LayerNorm's `rsqrt(var + eps)`, where the two triangle-multiplication norms use AF2's
    `mean(x^2) - mean(x)^2` (`af2_reference.py:131`) and that form can go NEGATIVE under
    cancellation once the pad rows carry the outer product mean's 1/eps amplification;
  * every softmax's sum, which is zero on a fully masked row unless the max is subtracted. The
    installed ttnn 0.68.0 defaults both `ttnn.softmax` and `ttnn.softmax_in_place` to
    `numeric_stable=True`, and the reference subtracts the max, so this one is recorded as a
    count of fully masked rows rather than as a suspect.

bfloat16 has float32's exponent range, so a float32 magnitude is the right overflow test for
both: a bf16 value overflows exactly where the fp32 one exceeds ~3.39e38. The LayerNorm variance
is ALSO taken from bf16-rounded inputs with fp32 accumulation, which is the device's storage and
accumulate precision, because cancellation is the one failure bf16 reaches before fp32 does.

Regions: rr is a real-real residue pair, rp touches exactly one pad residue, pp is pad-pad.

Three parts, each written to the JSON as it lands:
  A. BindCraft 2's own bucket-32 state: forward stats per op per region for `af2`,
     `device_today` and `device_fixed`, then a backward with one seeded cotangent on the real
     positions, the pad zero-filled as `splice.py:_backward` does.
  B. The regime the live loop runs: bucket 1 padded internally by P residues exactly as
     `splice.py:_pad_inputs` does, P in 0 / 13 / 32 / 64, distance on the real block against the
     P=0 `af2` output. The five device trajectories all ran n = 19 mod 32, so every one of them
     padded by exactly 13 there, which is why a pad-driven defect would read FLAT across them.
  C. `pairmask_grade`'s four arms against BindCraft 2's JAX, bucket 32 and bucket 1.
"""
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

import pairmask_grade as G                                              # noqa: E402

OUT = HERE / "finite_probe.json"
EPS_LN = 1e-5
#: 48 is the model. Fewer is only for the smoke test, which checks the plumbing, not a number.
BLOCKS = int(os.environ.get("BCX_BLOCKS", "48"))


def regions(seq: torch.Tensor) -> dict:
    s = seq.bool()
    rr = s[:, None] & s[None, :]
    pp = ~s[:, None] & ~s[None, :]
    return {"rr": rr, "rp": ~rr & ~pp, "pp": pp}


class Stats:
    """Per-op, per-region: non-finite count and max |x|; per-LayerNorm: the variance margin."""

    def __init__(self, seq: torch.Tensor):
        self.seq = seq.bool()
        self.reg = regions(seq)
        self.ops: dict = {}
        self.ln: dict = {}
        self.on = True

    def _masks_for(self, t: torch.Tensor):
        n = self.seq.numel()
        if t.dim() == 3 and t.shape[0] == n and t.shape[1] == n:
            return self.reg
        if t.dim() == 3 and t.shape[1] == n:               # [rows, n, c]
            return {"r": self.seq[None, :].expand(t.shape[0], n),
                    "p": ~self.seq[None, :].expand(t.shape[0], n)}
        if t.dim() == 3 and t.shape[0] == n:               # [n, rows, c]
            return {"r": self.seq[:, None].expand(n, t.shape[1]),
                    "p": ~self.seq[:, None].expand(n, t.shape[1])}
        return {"all": torch.ones(t.shape[:-1], dtype=torch.bool)}

    def op(self, name: str, t: torch.Tensor):
        if not self.on:
            return
        t = t.detach()
        e = self.ops.setdefault(name, {})
        for r, m in self._masks_for(t).items():
            x = t[m]
            if x.numel() == 0:
                continue
            d = e.setdefault(r, {"nonfinite": 0, "max_abs": 0.0})
            d["nonfinite"] += int((~torch.isfinite(x)).sum())
            fin = x[torch.isfinite(x)]
            if fin.numel():
                d["max_abs"] = max(d["max_abs"], float(fin.abs().max()))

    def layernorm(self, name: str, x: torch.Tensor):
        if not self.on:
            return
        x = x.detach().float()
        xb = x.bfloat16().float()
        fast = (xb * xb).mean(-1) - xb.mean(-1) ** 2          # bf16 in, fp32 accumulate
        exact = x.double().var(-1, unbiased=False)
        e = self.ln.setdefault(name, {})
        for r, m in self._masks_for(x).items():
            if not bool(m.any()):
                continue
            d = e.setdefault(r, {"min_fast_var": float("inf"), "min_exact_var": float("inf"),
                                 "fast_var_plus_eps_le_0": 0, "max_mean_sq_over_var": 0.0})
            f, v, mu = fast[m], exact[m], x.double().mean(-1)[m]
            d["min_fast_var"] = min(d["min_fast_var"], float(f.min()))
            d["min_exact_var"] = min(d["min_exact_var"], float(v.min()))
            d["fast_var_plus_eps_le_0"] += int((f + EPS_LN <= 0).sum())
            ratio = (mu * mu / v.clamp_min(1e-300))
            d["max_mean_sq_over_var"] = max(d["max_mean_sq_over_var"], float(ratio.max()))


def install_layernorm_hook(stats_ref: list):
    """Every AF2 LayerNorm reports its input before it normalises it."""
    import tt_bio.af2_reference as R
    real = R.LayerNorm.forward

    def forward(self, x):
        st = stats_ref[0]
        if st is not None:
            st.layernorm("fast" if self.fast_variance else "exact", x)
        return real(self, x)
    R.LayerNorm.forward = forward


def arm_masks(mode: str, pm: torch.Tensor):
    """(multiply mask, start-attention mask, end-attention mask, one-sided multiply?)

    `device_fixed` is what `af2.af2_pair_masks` hands the device: a `[1, 1, 1, n]` key bias,
    the same tensor for both attentions, because the device's ending variant transposes the pair
    and then adds the bias on the key axis. The reference transposes its MASK instead
    (`af2_reference.py:261`), so the ending attention is given the transpose here, and after the
    reference undoes it every query row sees `seq_mask` on its keys -- which is what the card does.
    """
    ones = torch.ones_like(pm)
    seq = torch.diagonal(pm)
    if mode == "device_today":
        return ones, ones, ones, False
    if mode == "device_fixed":
        return pm, ones * seq[None, :], seq[:, None] * ones, True
    return pm, pm, pm, False


def run_stack(model, m, z, mm, pm, mode, stats=None):
    """48 blocks, op by op, with the arm's masks.

    Differentiable if the inputs are, and then each block is checkpointed: an uncheckpointed
    fp32 backward over 48 blocks at n=211 holds tens of GB of attention logits. The recompute
    runs after this function has restored the both-halves multiplication, so only the arms whose
    multiplication IS both-halves (`af2`, `device_today`) may be differentiated.
    """
    import tt_bio.af2_reference as R
    from torch.utils.checkpoint import checkpoint
    pm_mul, pm_start, pm_end, one = arm_masks(mode, pm)
    taped = torch.is_grad_enabled() and m.requires_grad
    assert not (taped and one), "a one-sided arm cannot be recomputed after the swap is undone"
    both = R.TriangleMultiplication.forward
    R.TriangleMultiplication.forward = G.one_sided_trimul() if one else both
    rec = (lambda name, t: stats.op(name, t)) if stats is not None else (lambda *_: None)

    def block(i, m, z):
        blk = model.evoformer[i]
        for name, fn in (("msa_row_attn", lambda: blk.msa_row_attn(m, mm, z)),
                         ("msa_col_attn", lambda: blk.msa_col_attn(m, mm)),
                         ("msa_transition", lambda: blk.msa_transition(m))):
            u = fn(); rec(name, u); m = m + u
        rec("msa", m)
        for name, fn in (("opm", lambda: blk.opm(m, mm)),
                         ("tri_mul_out", lambda: blk.tri_mul_out(z, pm_mul)),
                         ("tri_mul_in", lambda: blk.tri_mul_in(z, pm_mul)),
                         ("tri_att_start", lambda: blk.tri_att_start(z, pm_start)),
                         ("tri_att_end", lambda: blk.tri_att_end(z, pm_end)),
                         ("pair_transition", lambda: blk.pair_transition(z))):
            u = fn(); rec(name, u); z = z + u
        rec("pair", z)
        return m, z

    try:
        for i in range(BLOCKS):
            m, z = (checkpoint(block, i, m, z, use_reentrant=False) if taped
                    else block(i, m, z))
    finally:
        R.TriangleMultiplication.forward = both
    return m, z


def tensors(cap):
    f = lambda k: torch.from_numpy(np.asarray(cap[k])).float()
    return f("msa_in"), f("pair_in"), f("msa_mask"), f("pair_mask")


def part_a(ref, cap, out):
    model = ref["f32"]
    m0, z0, mm, pm = tensors(cap)
    seq = torch.diagonal(pm)
    n, n_pad = int(seq.numel()), int((seq == 0).sum())
    fully_masked_cols = int((mm.sum(0) == 0).sum())
    res = {"n": n, "n_pad": n_pad, "pad_at": torch.nonzero(seq == 0).flatten().tolist(),
           "msa_mask_rows": int(mm.shape[0]),
           "msa_mask_row0_equals_seq_mask": bool(torch.equal(mm[0], seq)),
           "residues_with_every_msa_row_masked": fully_masked_cols,
           "opm_pairs_at_eps_divisor": int(n * n - int((seq[:, None] * seq[None, :]).sum())),
           "opm_bias_over_eps_per_block_max": float(max(
               float(model.evoformer[i].opm.proj_o.bias.abs().max()) for i in range(48)) / 1e-3),
           "arms": {}}
    stats_ref = [None]
    install_layernorm_hook(stats_ref)
    torch.manual_seed(0)
    real_m = seq.bool()[None, :, None]
    real_z = (seq[:, None] * seq[None, :]).bool()[..., None]
    for mode in ("af2", "device_today", "device_fixed"):
        t0 = time.time()
        st = Stats(seq)
        stats_ref[0] = st
        with torch.no_grad():
            run_stack(model, m0.clone(), z0.clone(), mm, pm, mode, st)
        stats_ref[0] = None
        arm = {"forward_s": round(time.time() - t0, 1), "ops": st.ops, "layernorm": st.ln}
        if mode in ("af2", "device_today"):
            t0 = time.time()
            ml, zl = m0.clone().requires_grad_(True), z0.clone().requires_grad_(True)
            mo, zo = run_stack(model, ml, zl, mm, pm, mode)
            g = torch.Generator().manual_seed(1)
            cm = torch.randn(mo.shape, generator=g) * real_m
            cz = torch.randn(zo.shape, generator=g) * real_z
            ((mo * cm).sum() + (zo * cz).sum()).backward()
            gst = Stats(seq)
            gst.op("d_msa_in", ml.grad)
            gst.op("d_pair_in", zl.grad)
            arm["backward"] = gst.ops
            arm["backward_s"] = round(time.time() - t0, 1)
        res["arms"][mode] = arm
        print(f"A {mode}: fwd {arm['forward_s']} s", flush=True)
        out["A_bucket32_finiteness"] = res
        OUT.write_text(json.dumps(out, indent=1, default=float))


def pad_like_splice(m, z, mm, pad):
    """`splice.py:_pad_inputs`: zero-fill the token axis and mask the pad in the MSA mask only.
    The pair mask is what the pair track WOULD need; today's device never receives it."""
    F = torch.nn.functional
    if not pad:
        return m, z, mm, torch.ones(z.shape[0], z.shape[0])
    m = F.pad(m, (0, 0, 0, pad)); z = F.pad(z, (0, 0, 0, pad, 0, pad))
    mm = F.pad(mm, (0, pad))
    seq = torch.ones(z.shape[0]); seq[-pad:] = 0
    return m, z, mm, seq[:, None] * seq[None, :]


def part_b(ref, cap, out):
    model = ref["f32"]
    m0, z0, mm0, _ = tensors(cap)
    n = int(z0.shape[0])
    single = model.single_activations

    def cmp(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        return {"rel_l2": float((a - b).norm() / b.norm()),
                "cos": float(a @ b / (a.norm() * b.norm()))}

    with torch.no_grad():
        base_m, base_z = run_stack(model, m0.clone(), z0.clone(), mm0, torch.ones(n, n), "af2")
        base_s = single(base_m[0])
        res = {"n_real": n, "reference": "af2 arm, P=0, same fp32 code", "pads": {}}
        for pad in (13, 32, 64):
            m, z, mm, pm = pad_like_splice(m0, z0, mm0, pad)
            row = {}
            for mode in ("af2", "device_today", "device_fixed"):
                t0 = time.time()
                mo, zo = run_stack(model, m.clone(), z.clone(), mm, pm, mode)
                mo, zo = mo[:, :n], zo[:n, :n]
                row[mode] = {"msa": cmp(mo, base_m), "pair": cmp(zo, base_z),
                             "single": cmp(single(mo[0]), base_s),
                             "nonfinite": int((~torch.isfinite(zo)).sum())
                             + int((~torch.isfinite(mo)).sum())}
                print(f"B pad {pad} {mode}: pair {row[mode]['pair']['rel_l2']:.3e} "
                      f"({time.time() - t0:.0f} s)", flush=True)
            res["pads"][str(pad)] = row
            out["B_pad_sweep_bucket1_internal_pad"] = res
            OUT.write_text(json.dumps(out, indent=1, default=float))


def part_c(ref, caps, out):
    res = {}
    for bucket, cap in caps.items():
        arms = G.run_arms(cap, ref)
        res[f"bucket_{bucket}"] = {"n": int(cap["pair_in"].shape[0]),
                                   "n_masked": int((np.diagonal(cap["pair_mask"]) == 0).sum()),
                                   "arms": G.grade(cap, arms, ref)}
        out["C_vs_bc2_jax"] = res
        OUT.write_text(json.dumps(out, indent=1, default=float))
        print(f"C bucket {bucket} graded", flush=True)


def main():
    import afgrad as A
    _, ref = A.load_models(G.PARAM_NPZ, device_arm=False)
    out = json.loads(OUT.read_text()) if OUT.is_file() else {}
    cap32 = G.cached_capture(32)
    if "A_bucket32_finiteness" not in out or len(out["A_bucket32_finiteness"]["arms"]) < 3:
        part_a(ref, cap32, out)
    cap1 = G.cached_capture(1)
    if "B_pad_sweep_bucket1_internal_pad" not in out or \
            len(out["B_pad_sweep_bucket1_internal_pad"]["pads"]) < 3:
        part_b(ref, cap1, out)
    part_c(ref, {32: cap32, 1: cap1}, out)
    print("finite_probe complete", flush=True)


if __name__ == "__main__":
    main()
