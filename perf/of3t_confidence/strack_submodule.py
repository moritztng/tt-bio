"""Which sub-module of the device s-track disagrees: AttentionPairBias or the Transition?

Both are fed the host block's exact input and compared against the host formula in
`_host_s_block`. The device s-track has never been exercised for this head -- the weights
were remapped and the modules built, but the shipped head always ran the host s-path -- so
"never gated" is the prior here, not "bf16".
"""
import os, math, pickle, sys, torch, torch.nn.functional as F, ttnn
sys.path.insert(0, os.path.expanduser("~/.coworker/wt/of3t-confidence/perf/of3t_confidence"))
from forward_vs_float64 import ca_walk, rel_l2
from tt_bio.tenstorrent import get_device
from tt_bio.openfold3_confidence import OF3ConfidenceHead

_C_S, _C_Z, _H, _HD = 384, 128, 16, 24
CK = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
sd = torch.load(CK, map_location="cpu", weights_only=False)
if hasattr(sd, "state_dict"): sd = sd.state_dict()
aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
g = pickle.load(open(os.path.expanduser("~/of3_ref_out.pkl"), "rb"))["intermediates"]
si_input = g["input_embedder_real"]["out"][0].float()
si_trunk, zij_trunk = (t.float() for t in g["pairformer_stack_real"]["out"])
N = si_trunk.shape[0]
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
head = OF3ConfidenceHead(aux, dev, ckc)
steps = []
orig = OF3ConfidenceHead._host_s_block
def spy(self, s, z, i):
    o = orig(self, s, z, i); steps.append((s.clone(), z.clone(), o.clone())); return o
OF3ConfidenceHead._host_s_block = spy
head.forward(si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, repr_x_pred=ca_walk(N),
             max_atom_per_token_mask=torch.ones(N * 23), use_zij_trunk_embedding=True, s_path="host")
OF3ConfidenceHead._host_s_block = orig
head._dtype = torch.float32
up = lambda x: ttnn.from_torch(x.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)
dn = lambda t, *sh: torch.Tensor(ttnn.to_torch(t)).float().reshape(*sh)

print(f"{'blk':>3s} {'APB o relL2':>12s} {'APB scaled':>11s} {'best 1/sqrt(d)':>15s} "
      f"{'transition relL2':>17s}")
for i, (s_in, z_in, _) in enumerate(steps):
    blk = head.pf.blocks[i]
    bw = lambda n: head._bw(i, "attn_pair_bias." + n)
    a = F.layer_norm(s_in, (_C_S,), bw("layer_norm_a.weight"), bw("layer_norm_a.bias"))
    # host AttentionPairBias output `o`
    zn = F.layer_norm(z_in, (_C_Z,), bw("layer_norm_z.weight"), bw("layer_norm_z.bias"))
    bias = F.linear(zn, bw("linear_z.weight")).permute(2, 0, 1)
    q = F.linear(a, bw("mha.linear_q.weight"), bw("mha.linear_q.bias"))
    k = F.linear(a, bw("mha.linear_k.weight")); v = F.linear(a, bw("mha.linear_v.weight"))
    qh = q.view(N, _H, _HD).permute(1, 0, 2); kh = k.view(N, _H, _HD).permute(1, 0, 2)
    vh = v.view(N, _H, _HD).permute(1, 0, 2)
    gate = torch.sigmoid(F.linear(a, bw("mha.linear_g.weight")))
    def host_o(scale):
        sc = torch.einsum("hqd,hkd->hqk", qh * scale, kh) + bias
        o = torch.einsum("hqk,hkd->hqd", F.softmax(sc, -1), vh)
        o = o.permute(1, 0, 2).reshape(N, _H * _HD)
        return F.linear(o * gate, bw("mha.linear_o.weight"))
    ref = host_o(_HD ** -0.5)
    got = dn(blk.attention_pair_bias(up(a), up(z_in)), N, _C_S)
    alt = rel_l2(got, host_o(32 ** -0.5))
    best = min(range(8, 65), key=lambda d: rel_l2(got, host_o(d ** -0.5)))
    # transition_s on the post-APB s
    s2 = s_in + ref
    tp = "single_transition."
    xn = F.layer_norm(s2, (_C_S,), head._bw(i, tp + "layer_norm.weight"),
                      head._bw(i, tp + "layer_norm.bias"))
    t_ref = F.linear(F.silu(F.linear(xn, head._bw(i, tp + "swiglu.linear_a.weight")))
                     * F.linear(xn, head._bw(i, tp + "swiglu.linear_b.weight")),
                     head._bw(i, tp + "linear_out.weight"))
    t_got = dn(blk.transition_s(up(s2)), N, _C_S)
    print(f"{i:3d} {rel_l2(got, ref):12.3e} {alt:11.3e} {best:15d} {rel_l2(t_got, t_ref):17.3e}")
