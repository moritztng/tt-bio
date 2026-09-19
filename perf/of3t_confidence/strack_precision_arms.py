"""Which part of the bf16 s-track costs pLDDT: the residual, the weights, or the attention?

Runs the confidence head's host s-path on the REAL per-block z (captured from the device
z-track, so the z-path is identical across arms) and rounds exactly one group of tensors to
bf16 per arm. The comparand is LN(s) and the pLDDT logits, because those are what the head
reads -- a relative L2 on s itself is dominated by a 2.3e5 residual that every arm shares.
"""
import os, math, pickle, sys, torch, torch.nn.functional as F, ttnn

sys.path.insert(0, os.path.expanduser("~/.coworker/wt/of3t-confidence/perf/of3t_confidence"))
from forward_vs_float64 import ca_walk, rel_l2
from tt_bio.tenstorrent import get_device
from tt_bio.openfold3_confidence import OF3ConfidenceHead

CK = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
G = os.path.expanduser("~/of3_ref_out.pkl")
_C_S, _C_Z, _H, _HD = 384, 128, 16, 24
BF = lambda x: x.to(torch.bfloat16).float()
ID = lambda x: x


def s_block(head, s, z_host, i, r_res=ID, r_w=ID, r_act=ID):
    """`_host_s_block` with three rounding hooks: the residual, the weights, the activations."""
    bw = lambda n: r_w(head._bw(i, n))
    pfx = "attn_pair_bias."
    a = r_act(F.layer_norm(s, (_C_S,), bw(pfx + "layer_norm_a.weight"), bw(pfx + "layer_norm_a.bias")))
    zn = r_act(F.layer_norm(z_host, (_C_Z,), bw(pfx + "layer_norm_z.weight"), bw(pfx + "layer_norm_z.bias")))
    bias = r_act(F.linear(zn, bw(pfx + "linear_z.weight")).permute(2, 0, 1))
    q = r_act(F.linear(a, bw(pfx + "mha.linear_q.weight"), bw(pfx + "mha.linear_q.bias")))
    k = r_act(F.linear(a, bw(pfx + "mha.linear_k.weight")))
    v = r_act(F.linear(a, bw(pfx + "mha.linear_v.weight")))
    N = a.shape[0]
    q = q.view(N, _H, _HD).permute(1, 0, 2) / math.sqrt(_HD)
    k = k.view(N, _H, _HD).permute(1, 0, 2)
    v = v.view(N, _H, _HD).permute(1, 0, 2)
    scores = r_act(torch.einsum("hqd,hkd->hqk", q, k) + bias)
    o = r_act(torch.einsum("hqk,hkd->hqd", F.softmax(scores, dim=-1), v))
    o = o.permute(1, 0, 2).reshape(N, _H * _HD)
    g = torch.sigmoid(F.linear(a, bw(pfx + "mha.linear_g.weight")))
    o = r_act(F.linear(r_act(o * g), bw(pfx + "mha.linear_o.weight")))
    s = r_res(s + o)
    tp = "single_transition."
    xn = r_act(F.layer_norm(s, (_C_S,), bw(tp + "layer_norm.weight"), bw(tp + "layer_norm.bias")))
    t = r_act(F.silu(F.linear(xn, bw(tp + "swiglu.linear_a.weight"))) *
              F.linear(xn, bw(tp + "swiglu.linear_b.weight")))
    return r_res(s + F.linear(t, bw(tp + "linear_out.weight")))


def main():
    sd = torch.load(CK, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
    g = pickle.load(open(G, "rb"))["intermediates"]
    si_input = g["input_embedder_real"]["out"][0].float()
    si_trunk, zij_trunk = (t.float() for t in g["pairformer_stack_real"]["out"])
    N = si_trunk.shape[0]
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    head = OF3ConfidenceHead(aux, dev, ckc)
    zs = []
    orig = OF3ConfidenceHead._host_s_block
    OF3ConfidenceHead._host_s_block = lambda self, s, z, i: (zs.append(z.clone()) or orig(self, s, z, i))
    kw = dict(si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
              repr_x_pred=ca_walk(N), max_atom_per_token_mask=torch.ones(N * 23),
              use_zij_trunk_embedding=True)
    h = head.forward(**kw, s_path="host")
    OF3ConfidenceHead._host_s_block = orig
    dv = head.forward(**kw, s_path="device")
    head._dtype = torch.float32

    w = aux["plddt.layer_norm.weight"].float()
    b = aux.get("plddt.layer_norm.bias")
    b = b.float() if b is not None else 0.0
    L = aux["plddt.linear.weight"].float()
    plddt = lambda s: F.linear(F.layer_norm(s, (_C_S,)) * w + b, L)

    def run(**hooks):
        s = si_trunk.clone()
        for i in range(len(zs)):
            s = s_block(head, s, zs[i], i, **hooks)
        return s

    ref = run()
    print(f"host fp32 s-track reproduces forward(): s relL2 {rel_l2(ref, h['si_conf'].float()):.2e}")
    print(f"\n{'arm':34s} {'LN(s) relL2':>13s} {'plddt relL2':>13s}")
    arms = {
        "residual bf16": dict(r_res=BF),
        "weights bf16": dict(r_w=BF),
        "activations bf16": dict(r_act=BF),
        "all three bf16": dict(r_res=BF, r_w=BF, r_act=BF),
    }
    lnref = F.layer_norm(ref, (_C_S,))
    for name, hk in arms.items():
        s = run(**hk)
        print(f"  {name:32s} {rel_l2(F.layer_norm(s, (_C_S,)), lnref):13.3e} "
              f"{rel_l2(plddt(s), plddt(ref)):13.3e}")
    print(f"  {'DEVICE (measured)':32s} "
          f"{rel_l2(F.layer_norm(dv['si_conf'].float(), (_C_S,)), lnref):13.3e} "
          f"{rel_l2(dv['plddt_logits'].float(), plddt(ref)):13.3e}")


if __name__ == "__main__":
    main()
