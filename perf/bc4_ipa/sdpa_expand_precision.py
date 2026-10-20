"""usage: sdpa_expand_precision.py <dir holding ipa_phases.py> <params npz>

Price IPA attention as one SDPA: logits = q'.k' + bias', q' = [q, pw*qp], k' = [k, kp],
bias' = bias - pw/2 |kp|^2 + mask (the |qp|^2 row constant cancels in softmax). Emulates bf16
operands (q', k', v'; fp32 accumulate) against the explicit float64 point term (ipa_phases.py)."""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1]); import ipa_phases as G

def bf16(x):
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return b.astype(np.uint32).view(np.float32).astype(np.float64)

rng = np.random.default_rng(3)
p = {k: np.asarray(v, np.float64)
     for k, v in np.load(sys.argv[2]).items() if "structure_module" in k}
p = {k.replace("alphafold/alphafold_iteration/structure_module/fold_iteration/", G.SM): v for k, v in p.items()}
n = 224
for scale in (0.5, 1.5, 3.0):                       # translation spread in internal units (x20 A)
    act = rng.standard_normal((n, G.CS)); z = rng.standard_normal((n, n, G.CZ))
    q = rng.standard_normal((n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    rot = G.quaternion_rotation(q[:, 1], q[:, 2], q[:, 3])
    trans = rng.standard_normal((n, 3)) * scale
    mask = np.ones(n)
    r = G.row_phase_in(act, rot, trans, p)
    bias = G.linear(z, p, G.SM + "invariant_point_attention/attention_2d")
    raw = p[G.SM + "invariant_point_attention//trainable_point_weights"]
    pw = np.sqrt(1.0 / (G.PQK * 9.0 / 2)) * np.logaddexp(raw, 0.0)
    d2 = ((r["qp"][:, None] - r["kp"][None]) ** 2).sum(-1)
    exact = (-0.5 * (pw[:, None] * d2).sum(-1) + np.einsum("qhc,khc->qkh", r["q"], r["k"]) + bias) / np.sqrt(3)
    for rnd, name in ((lambda x: x, "fp32-exact operands"), (bf16, "bf16 operands")):
        qa = np.concatenate([r["q"], pw[None, :, None] * r["qp"].reshape(n, 12, -1)], -1)   # [N,H,28]
        ka = np.concatenate([r["k"], r["kp"].reshape(n, 12, -1)], -1)
        ka2 = (r["kp"] ** 2).sum((-1, -2))                                                # [N,H]
        lg = (np.einsum("qhc,khc->qkh", rnd(qa), rnd(ka)) + rnd(bias) - 0.5 * pw * ka2[None]) / np.sqrt(3)
        e_l = lg - lg.max(1, keepdims=True); ex = exact - exact.max(1, keepdims=True)
        # the row constant -pw/2|qp|^2 is dropped from lg: compare softmaxes, not raw logits
        a1 = np.exp(e_l) / np.exp(e_l).sum(1, keepdims=True); a0 = np.exp(ex) / np.exp(ex).sum(1, keepdims=True)
        o1 = np.einsum("qkh,khc->qhc", a1, rnd(r["v"])); o0 = np.einsum("qkh,khc->qhc", a0, r["v"])
        lg2 = (-0.5 * (pw[:, None] * d2).sum(-1) + np.einsum("qhc,khc->qkh", rnd(r["q"]), rnd(r["k"])) + rnd(bias)) / np.sqrt(3)
        e2 = lg2 - lg2.max(1, keepdims=True); a2 = np.exp(e2) / np.exp(e2).sum(1, keepdims=True)
        print(f"   explicit fp32 point term, {name} scalar q/k/bias: attn rel L2 {np.linalg.norm(a2-a0)/np.linalg.norm(a0):.2e}")
        print(f"trans x{scale:3.1f}  {name:20s} attn rel L2 {np.linalg.norm(a1-a0)/np.linalg.norm(a0):.2e}"
              f"  attn@v rel L2 {np.linalg.norm(o1-o0)/np.linalg.norm(o0):.2e}  max|kp|^2 {ka2.max():.1f}")
