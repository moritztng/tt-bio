"""Decisive CPU test: does the device's bias pre-scaling explain the tri_att error?

The device TriangleAttention pre-scales its triangle bias by sqrt(head_dim):
    bias_weight = linear_z.weight * head_dim**0.5
    SDPA: scores = q@k^T * head_dim**-0.5 + bias(=linear_z*sqrt(head_dim))
But the OF3 + Protenix references add linear_z to the scores UNSCALED:
    scores = q@k^T / sqrt(head_dim) + linear_z   (q pre-divided by sqrt in _prep_qkv)
So the device over-weights the triangle bias by sqrt(head_dim). This script runs the
reference TriangleAttention (tri_att_start) in fp32 with the bias UNSCALED (correct) vs
scaled by sqrt(head_dim) (device behavior), and compares both to the correct fp32
output. If the scaled-bias version's PCC ~= the device's 0.990 update PCC, the over-
weighting is the lever. Also reports the bias magnitude vs the q@k^T score magnitude to
show why it bites at OF3's regime.

    OF3_REF=/tmp/of3-ref TT_BIO_ROOT=<worktree> /tmp/of3-venv/bin/python scripts/of3_triatt_bias_scale_test.py
"""
import os, sys, math, pickle
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.expanduser(
    "~/.coworker/wt/tt-bio-openfold3-msa-precision-gap"))
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
QUERY_JSON = os.path.join(OF3_REF, "examples/example_inference_inputs/query_ubiquitin.json")
GOLD = os.path.expanduser("~/of3_msa_pair_stack_subops.pkl")
OUT = os.path.expanduser("~/of3_triatt_bias_scale.pkl")
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)


def sub(sd, prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _kw():
    return dict(chunk_size=None, transition_ckpt_chunk_size=None,
                use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False,
                _mask_trans=True)


def main():
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.core.model.feature_embedders.input_embedders import (
        InputEmbedderAllAtom, MSAModuleEmbedder)
    from openfold3.core.model.latent.msa_module import MSAModuleBlock
    from openfold3.core.model.layers.triangular_attention import TriangleAttention
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet)
    from tt_bio.openfold3_data import build_openfold3_features
    import inspect

    g = pickle.load(open(GOLD, "rb"))
    z_in_pair = g["z_in"]               # z into pair_stack [N,N,c]
    z_tri_att_in = g["fp32"]["tri_mul_in"]   # z into tri_att_start [N,N,c]

    qs = InferenceQuerySet.from_json(QUERY_JSON)
    query = next(iter(qs.queries.values()))
    feat = build_openfold3_features(query)
    batch = {k: v.unsqueeze(0) for k, v in feat.items() if torch.is_tensor(v)}
    torch.manual_seed(0)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)

    # Build block-0 tri_att_start reference
    MSA_PF = dict(C.architecture.msa.msa_module)
    params = set(inspect.signature(MSAModuleBlock.__init__).parameters) - {"self", "last_block"}
    blk0 = MSAModuleBlock(**{k: v for k, v in MSA_PF.items() if k in params}, last_block=False).eval()
    blk0.load_state_dict(sub(sd, "msa_module.blocks.0"), strict=True)
    tri_att_start = blk0.pair_stack.tri_att_start  # starting=True, c_hidden=32, heads=4

    c_hidden = tri_att_start.mha.c_hidden  # per-head = 32
    sqrt_d = math.sqrt(c_hidden)
    print(f"c_hidden(per-head)={c_hidden} sqrt={sqrt_d:.4f}")

    z = z_tri_att_in.unsqueeze(0).float()  # [1,N,N,c]
    pair_mask = g["pair_mask"].unsqueeze(0).float()
    with torch.no_grad():
        # Correct (unscaled bias) -- this IS the fp32 reference output
        o_correct = tri_att_start(z, mask=pair_mask, chunk_size=None)
        # Device behavior: over-weight the triangle bias by sqrt(d). Patch linear_z weight.
        orig_w = tri_att_start.linear_z.weight.clone()
        tri_att_start.linear_z.weight.data = orig_w * sqrt_d
        o_scaled = tri_att_start(z, mask=pair_mask, chunk_size=None)
        tri_att_start.linear_z.weight.data = orig_w  # restore

    # The tri_att UPDATE = o (tri_att returns the update, not z+update)
    upd_correct = o_correct[0]
    upd_scaled = o_scaled[0]
    print(f"update pcc(scaled-bias vs correct) = {pcc(upd_scaled, upd_correct):.5f}  "
          f"(device tri_att_start update pcc was 0.99056)")

    # Bias vs score magnitude: compute the triangle_bias and the q@k^T/sqrt(d) scores
    with torch.no_grad():
        x = z
        x = x  # starting -> no transpose
        x_ln = tri_att_start.layer_norm(x)
        triangle_bias = tri_att_start.linear_z(x_ln)  # [1,N,N,H]
        H = triangle_bias.shape[-1]
        # reference adds triangle_bias (permuted) to scores q@k^T/sqrt(d)
        q = tri_att_start.mha.linear_q(x_ln); k = tri_att_start.mha.linear_k(x_ln)
        q = q.view(*q.shape[:-1], H, -1).transpose(-2, -3)
        k = k.view(*k.shape[:-1], H, -1).transpose(-2, -3)
        scores = torch.einsum("...qc,...kc->...qk", q / sqrt_d, k)  # [1,H,N,N]
        print(f"|q@k^T/sqrt(d)| std = {float(scores.std()):.4f}")
        tb = triangle_bias.permute(0, 3, 1, 2)  # [1,H,N,N]
        print(f"|triangle_bias| std = {float(tb.std()):.4f}  (device over-weights x{sqrt_d:.2f} "
              f"-> {float((tb*sqrt_d).std()):.4f})")
        print(f"ratio bias/score (correct) = {float(tb.std()/scores.std()):.4f}")
        print(f"ratio bias/score (device)   = {float((tb*sqrt_d).std()/scores.std()):.4f}")


if __name__ == "__main__":
    main()
