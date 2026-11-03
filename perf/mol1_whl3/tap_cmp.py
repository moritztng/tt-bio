"""Grade WH's trunk block 43 single Transition (tap.pt from the WHL3_TAP cell) op by op against upstream fp32 (split.pt)
and against each op recomputed in float64 on WH's OWN input (the op's own error, inherited error removed).

    python -I tap_cmp.py CKPT SPLIT.pt TAP.pt UP_LAYERS.pt
Columns: token-part rel err on the small output channels (or all channels for 1536-wide intermediates)."""
import sys, torch
import torch.nn.functional as F
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
sd = ck.get("model", ck.get("state_dict", ck))
pre = "module.pairformer_stack.blocks.43.single_transition."
lw, lb, wa, wb, wo = (sd[pre + k].double() for k in ("layernorm1.weight", "layernorm1.bias", "linear_no_bias_a.weight",
                                                      "linear_no_bias_b.weight", "linear_no_bias.weight"))
r = torch.load(sys.argv[2], weights_only=False); tap = torch.load(sys.argv[3], weights_only=False)
up = torch.load(sys.argv[4], weights_only=False)
n = r["tr.in"][0].reshape(-1, 384).shape[0]
small = up[-1].abs().amax(0) < 512
print("WH ckc:", tap.get("ckc")); print("WH dtype:", tap.get("dtype"))
print("WH captured:", {k: [tuple(t.shape) for t in v] for k, v in tap.items() if isinstance(v, list)})
def get(k):
    v = tap[k]; c = v[0].shape[-1]
    return torch.cat([t.reshape(-1, c) for t in v]).double()[:n]
def tok(t): return t - t.mean(0)
def rel(a, b): return float((a - b).norm() / b.norm())
def score(a, b): return rel(tok(a[:, small]), tok(b[:, small])) if a.shape[-1] == 384 else rel(tok(a), tok(b))
U = {"in": r["tr.in"][0], "layer_norm": r["tr.ln.out"], "silu_a": F.silu(r["tr.a.out"].double()), "b": r["tr.b.out"],
     "hidden": r["tr.out.in"][0], "out": r["tr.out.out"]}
U = {k: v.double().reshape(-1, v.shape[-1]) for k, v in U.items()}
names = {"in": "in", "layer_norm": "layer_norm", "silu_a": "_transition_linear:fc1", "b": "_transition_linear:fc2",
         "hidden": next((k for k in ("multiply_", "multiply") if k in tap), None), "out": "out"}
W = {k: get(v) for k, v in names.items() if v in tap}
own = {}
if "in" in W: own["layer_norm"] = F.layer_norm(W["in"], (384,), lw, lb, 1e-5)
if "layer_norm" in W: own["silu_a"] = F.silu(W["layer_norm"] @ wa.T); own["b"] = W["layer_norm"] @ wb.T
if "silu_a" in W and "b" in W: own["hidden"] = W["silu_a"] * W["b"]
if "hidden" in W: own["out"] = W["hidden"] @ wo.T
for k in U:
    if k not in W:
        print(f"{k:10s} not captured"); continue
    line = f"{k:10s} vs upstream {score(W[k], U[k]):8.3g}"
    if k in own:
        line += f"   own-op err {score(W[k], own[k]):8.3g}   fp64 on WH input vs upstream {score(own[k], U[k]):8.3g}"
    print(line)
