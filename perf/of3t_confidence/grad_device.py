"""PROTOCOL SS3d: device gradients per parameter against the validated float64 reference.

Run `grad_reference.py` first; it writes the reference gradients and the seed.

Bijection (SS3a). Every device weight is recorded at upload time together with the
transform that produced it, by wrapping `Module.torch_to_tt` for the duration of the
head's construction. The gradient is then mapped back into THEIR space by inverting that
transform, never by applying it forward, and anything whose transform cannot be inverted
exactly is listed as unmapped with its reason rather than quietly dropped.
"""
import math, os, pickle, sys, torch, ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forward_vs_float64 import ca_walk
from tt_bio import autograd as ag
from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import get_device
from tt_bio.openfold3_confidence import OF3ConfidenceHead

HEADS = ["plddt_logits", "experimentally_resolved_logits", "pae_logits",
         "pde_logits", "distogram_logits"]
_BLK = "pairformer_embedding.pairformer_stack.blocks.%d."
S = "/tmp/of3t/of3t-confidence"

# The device module tree against THEIR names. Written out rather than derived, because a
# derivation that guessed wrong would silently compare two different tensors -- and a
# per-parameter claim over a map nobody checked is the defect SS3a exists to prevent.
_ZSUB = {"triangle_multiplication_start": "tri_mul_out",
         "triangle_multiplication_end": "tri_mul_in",
         "triangle_attention_start": "tri_att_start",
         "triangle_attention_end": "tri_att_end",
         "transition_z": "transition_z"}
_APB_KEY = {"proj_q.weight": "mha.linear_q.weight", "proj_q.bias": "mha.linear_q.bias",
            "proj_k.weight": "mha.linear_k.weight", "proj_v.weight": "mha.linear_v.weight",
            "proj_g.weight": "mha.linear_g.weight", "proj_o.weight": "mha.linear_o.weight",
            "proj_z.0.weight": "layer_norm_z.weight", "proj_z.0.bias": "layer_norm_z.bias",
            "proj_z.1.weight": "linear_z.weight"}
_TS_KEY = {"fc1.weight": "swiglu.linear_a.weight", "fc2.weight": "swiglu.linear_b.weight",
           "fc3.weight": "linear_out.weight", "norm.weight": "layer_norm.weight",
           "norm.bias": "layer_norm.bias"}


def their_name(path, key):
    """(device path under head.pf, scope-local key) -> (where, their name) or None."""
    parts = path.split(".")
    if len(parts) < 3 or parts[0] != "blocks":
        return None
    i, sub = int(parts[1]), parts[2]
    if sub in _ZSUB:                       # the z-track lives in the reference blocks
        # TriangleAttention keeps its q/k/v/g/o under `mha.`; its layer_norm and its
        # bias projection do not. Without this the five projection weights of every
        # triangle attention drop out of the comparison silently, which is 16 tensors
        # and exactly the kind of quiet hole SS3a's manifest exists to make impossible.
        if sub.startswith("triangle_attention") and key.split(".")[0] in (
                "linear_q", "linear_k", "linear_v", "linear_g", "linear_o"):
            key = "mha." + key
        return ("blk", i, f"{_ZSUB[sub]}.{key}",
                f"{_BLK % i}pair_stack.{_ZSUB[sub]}.{key}"
                if sub != "transition_z" else f"{_BLK % i}pair_stack.pair_transition.{key}")
    if sub == "attention_pair_bias" and key in _APB_KEY:
        n = f"{_BLK % i}attn_pair_bias.{_APB_KEY[key]}"
        return ("aux", i, n, n)
    if sub == "transition_s" and key in _TS_KEY:
        n = f"{_BLK % i}single_transition.{_TS_KEY[key]}"
        return ("aux", i, n, n)
    if sub.startswith("pre_norm_s"):       # PairformerLayer's own s LN is their layer_norm_a
        n = f"{_BLK % i}attn_pair_bias.layer_norm_a.{key.split('.')[-1]}"
        return ("aux", i, n, n)
    return None



def rel_l2(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a - b).norm() / (b.norm() + 1e-30))


def record_uploads():
    """Wrap Module.torch_to_tt so every device weight remembers where it came from."""
    orig = T.Module.torch_to_tt
    reg = {}

    def wrapped(self, key, transform=lambda x: x.t(), dtype=None):
        out = orig(self, key, transform, dtype)
        src = self.weights[key]
        reg[id(out)] = (self.weights, key, src, transform, out)
        return out

    T.Module.torch_to_tt = wrapped
    return reg, (lambda: setattr(T.Module, "torch_to_tt", orig))


def walk(obj, path="", seen=None, out=None):
    """Every ttnn.Tensor attribute reachable from the module tree, with its path."""
    seen = set() if seen is None else seen
    out = [] if out is None else out
    if id(obj) in seen:
        return out
    seen.add(id(obj))
    items = obj.items() if isinstance(obj, dict) else (
        enumerate(obj) if isinstance(obj, (list, tuple)) else
        getattr(obj, "__dict__", {}).items())
    for k, v in list(items):
        p = f"{path}.{k}" if path else str(k)
        if isinstance(v, ttnn.Tensor):
            out.append((p, obj, k, v))
        elif isinstance(v, (T.Module, dict, list, tuple)) or hasattr(v, "__dict__"):
            if not isinstance(v, (str, bytes, int, float, torch.Tensor)):
                walk(v, p, seen, out)
    return out


def main():
    sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                    weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
    g = pickle.load(open(os.path.expanduser("~/of3_ref_out.pkl"), "rb"))["intermediates"]
    si_input = g["input_embedder_real"]["out"][0].float()
    si_trunk, zij_trunk = (t.float() for t in g["pairformer_stack_real"]["out"])
    N = si_trunk.shape[0]
    repr_x, mask = ca_walk(N), torch.ones(N * 23)
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    reg, restore = record_uploads()
    head = OF3ConfidenceHead(aux, dev, ckc)
    restore()

    up = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(
        x.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    si_d = up(si_input)
    st_d = up(si_trunk, ttnn.float32)
    zt_d = up(zij_trunk)
    oh_d = head.distance_onehot(repr_x)
    # Warm `_wd` so every head/embedding weight exists before it is declared a parameter.
    head.forward_device(si_d, st_d, zt_d, oh_d)

    params = {}          # device tensor id -> (their-name, inverse_fn, leaf)
    for key, dev_t in list(head._wd_cache.items()):
        name, transposed, _dt = key
        leaf = ag.parameter(dev_t)
        head._wd_cache[key] = leaf
        params[name] = (leaf, (lambda x: x.t()) if transposed else (lambda x: x),
                        "aux", None, name)
    unmapped = []
    for path, owner, attr, t in walk(head.pf):
        rec = reg.get(id(t))
        if rec is None:
            unmapped.append((path, tuple(t.shape), "built inline, not via torch_to_tt"))
            continue
        weights, key, src, transform, _ = rec
        leaf = ag.parameter(t)
        setattr(owner, attr, leaf) if not isinstance(owner, dict) else owner.__setitem__(attr, leaf)
        probe = transform(src)
        if probe.shape == src.t().shape and torch.equal(probe, src.t()):
            inv = lambda x: x.t()
        elif probe.shape == src.shape and torch.equal(probe, src):
            inv = lambda x: x
        else:
            unmapped.append((path, tuple(t.shape), f"transform of {key} is not identity or "
                                                   f"transpose; inverse not established"))
            continue
        tn = their_name(path, key)
        if tn is None:
            unmapped.append((path, tuple(t.shape), f"no entry in the name map for {key}"))
            continue
        params[tn[3]] = (leaf, inv, tn[0], tn[1], tn[2])
    print(f"taped {len(params)} parameters; {len(unmapped)} device tensors unmapped")
    for p, s_, why in unmapped:
        print(f"  UNMAPPED {p:52s} {str(s_):22s} {why}")

    sti = ag.Tensor(st_d, requires_grad=True)
    zti = ag.Tensor(zt_d, requires_grad=True)
    with ag.tape():
        out = head.forward_device(ag.Tensor(si_d), sti, zti, ag.Tensor(oh_d))
    seed = torch.load(f"{S}/seed.pt")["seed"]
    roots, seeds = [], []
    for k in HEADS:
        o = out[k]
        sv = seed[k].reshape(tuple(int(d) for d in o.shape))
        roots.append(o)
        seeds.append(ttnn.from_torch(sv.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                     dtype=ttnn.bfloat16))
    ag.backward(roots, seeds)
    dn = lambda t: torch.Tensor(ttnn.to_torch(t)).double()
    ra = torch.load(f"{S}/ref_act_grad.pt")
    print("\nACTIVATION gradients -- the trunk connection:")
    for nm, leaf, ref in (("si_trunk", sti, ra["si_trunk"]), ("zij_trunk", zti, ra["zij_trunk"])):
        if leaf.grad is None:
            print(f"  {nm:12s} NO GRADIENT -- the graph is severed"); continue
        gd = dn(leaf.grad).reshape(ref.shape)
        print(f"  {nm:12s} relL2 {rel_l2(gd, ref):.4e}   ||ours|| {float(gd.norm()):.6g} "
              f"||ref|| {float(ref.norm()):.6g}")

    rg = torch.load(f"{S}/ref_aux_grad.pt")
    rb = torch.load(f"{S}/ref_blk_grad.pt")
    rows = []
    for their, (leaf, inv, where, bi, lookup) in sorted(params.items()):
        if leaf.grad is None:
            rows.append((their, None, "no gradient reached this leaf", 0.0, 0.0)); continue
        gd = inv(dn(leaf.grad))
        ref = rb[bi].get(lookup) if where == "blk" else rg.get(lookup)
        if ref is None:
            rows.append((their, None, f"no reference gradient under {lookup!r}", 0.0, 0.0)); continue
        if tuple(gd.shape) != tuple(ref.shape):
            rows.append((their, None, f"shape {tuple(gd.shape)} vs ref {tuple(ref.shape)}", 0.0, 0.0)); continue
        # SS3a: OF3's own checkpoint fuses TriangleMultiplication's a/b halves in p_in and
        # g_in, so each half is reported on its own. A fused number lets one half's
        # agreement mask the other half's error.
        if lookup.endswith(("p_in.weight", "g_in.weight")) and gd.shape[0] % 2 == 0:
            h = gd.shape[0] // 2
            rows.append((their + " [a]", rel_l2(gd[:h], ref[:h]), None,
                         float(ref[:h].norm()), float(gd[:h].norm())))
            rows.append((their + " [b]", rel_l2(gd[h:], ref[h:]), None,
                         float(ref[h:].norm()), float(gd[h:].norm())))
            continue
        rows.append((their, rel_l2(gd, ref), None, float(ref.norm()), float(gd.norm())))
    # A relative L2 against a STRUCTURALLY ZERO reference is not a measurement, it is a
    # division. `attn_pair_bias.layer_norm_z.bias` is the case here and it is exactly zero
    # by construction: that bias is per-channel on z, `linear_z` turns it into a per-head
    # constant, and a constant added to every key of a softmax row cancels. The reference
    # reads it at the float64 cancellation floor and ours at the bf16 one, so the ratio is
    # ~1e13 and means nothing. Reported as its own class with both absolute norms, never
    # folded into the statistics and never silently dropped.
    scale = sorted(r[3] for r in rows if r[1] is not None) or [1.0]
    med_ref = scale[len(scale) // 2]
    zerog = [r for r in rows if r[1] is not None and r[3] < 1e-8 * med_ref]
    rows = [r for r in rows if r not in zerog]
    ok = [(n, r) for n, r, e, _, _ in rows if r is not None]
    bad = [(n, e) for n, r, e, *_ in rows if r is None]
    ok.sort(key=lambda x: -x[1])
    print(f"\nPER-PARAMETER relative L2 vs float64 ({len(ok)} compared, {len(bad)} not):")
    for n, r in ok:
        print(f"  {'OVER' if r > 5e-2 else '    '} {n:64s} {r:.4e}")
    for n, e in bad:
        print(f"  SKIP {n:64s} {e}")
    for n, _r, _e, rn, gn in zerog:
        print(f"  ZERO {n:64s} ref |g|={rn:.3e}  ours |g|={gn:.3e} (exactly zero by "
              f"softmax invariance; relative L2 undefined)")
    if ok:
        med = sorted(r for _, r in ok)[len(ok) // 2]
        over = [n for n, r in ok if r > 5e-2]
        print(f"\nworst   {ok[0][0]}")
        print(f"        {ok[0][1]:.4e}   (per-tensor bar 5.0e-02)")
        print(f"median  {med:.4e}   (median bar 2.0e-02)")
        print(f"over the per-tensor bar: {len(over)}/{len(ok)}")
        pair = [r for n, r in ok if ".pair_stack." in n]
        rest = [r for n, r in ok if ".pair_stack." not in n]
        f = lambda v: f"max {max(v):.3e} median {sorted(v)[len(v)//2]:.3e}" if v else "none"
        print(f"  pair-track (z) parameters, {len(pair)}: {f(pair)}")
        print(f"  everything else,          {len(rest)}: {f(rest)}")


if __name__ == "__main__":
    main()
