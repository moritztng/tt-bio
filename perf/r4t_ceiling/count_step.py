#!/usr/bin/env python3
"""r4t-ceiling analytic count: FLOPs and DRAM bytes per op class of one RFD4 training step, counted on
upstream's own torch graph (CPU, float32) for a dumped crop-256 batch.

Phases, same protocol as r4t-perf PERF: K-candidate grad-off pass (K=4 rows, full recycle depth),
winner pass (K=1 row, upstream's own last-pass recycle grad), winner backward. Every aten op is counted:
FLOPs from torch's flop registry (matmul-class ops; others 1 flop/output element), bytes = inputs + outputs
as element counts (views and metadata ops = 0, they move nothing on the card either). Byte totals are
then priced at 2 B/element (bf16 activations) by the roofline step, not here.

    RFD4_REPO=... python count_step.py BATCH.npz OUT.json
"""
import json, os, sys, collections
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry
from torch.utils._pytree import tree_flatten

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r4t"))
import loss_parity_reference as L

A = torch.ops.aten
VIEW = {"view", "_unsafe_view", "expand", "t", "transpose", "permute", "unsqueeze", "squeeze", "select",
        "slice", "as_strided", "detach", "alias", "split", "split_with_sizes", "unbind", "chunk", "narrow",
        "lift_fresh", "_reshape_alias", "reshape", "diagonal", "unfold", "view_as_real", "view_as_complex",
        "empty", "empty_like", "zeros", "new_empty", "new_zeros", "empty_strided", "zeros_like", "ones_like",
        "full", "new_full", "scalar_tensor", "_local_scalar_dense", "is_same_size", "sym_size", "sym_stride",
        "set_", "resize_", "arange"}
MATMUL = {"mm", "bmm", "addmm", "baddbmm", "matmul", "linear", "_scaled_dot_product_flash_attention",
          "_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention_for_cpu",
          "convolution", "_scaled_dot_product_attention_math"}
NORM = {"native_layer_norm", "native_layer_norm_backward", "_fused_rms_norm", "rsqrt", "var_mean", "native_group_norm"}
MOVE = {"copy_", "_to_copy", "clone", "cat", "stack", "index", "index_put_", "index_select", "gather", "scatter",
        "scatter_add", "constant_pad_nd", "slice_backward", "select_backward", "index_add_", "masked_fill",
        "where", "repeat", "embedding", "embedding_dense_backward", "roll", "flip", "contiguous", "index_put",
        "_index_put_impl_", "new_zeros", "fill_", "zero_"}
PAIR_NUMEL = 256 * 256 * 8   # any tensor this big is pair-sized (K x I x I x C) or attention scores (H x I x I)

def ncls(name):
    if name in MATMUL: return "matmul"
    if name in NORM or "norm" in name: return "norm"
    if name in MOVE: return "move"
    if "softmax" in name: return "softmax"
    if name.startswith(("sum", "mean", "amax", "max", "min", "any", "all", "argmax", "argmin", "cumsum", "logsumexp", "norm")): return "reduce"
    return "eltwise"

class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.phase = "?"; self.t = collections.defaultdict(lambda: [0, 0, 0])  # flops, elems, calls
        self.ops = collections.Counter()
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func.overloadpacket.__name__
        if name in VIEW: return out
        ins = [a for a in tree_flatten((args, kwargs))[0] if isinstance(a, torch.Tensor)]
        outs = [o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor)]
        el = sum(x.numel() for x in ins) + sum(x.numel() for x in outs)
        big = max([x.numel() for x in ins + outs] + [0]) >= PAIR_NUMEL
        c = ncls(name)
        if func.overloadpacket in flop_registry:
            fl = flop_registry[func.overloadpacket](*args, **kwargs, out_val=out)
            if c not in ("matmul",): c = "matmul"
        else:
            fl = sum(o.numel() for o in outs)
        k = (self.phase, c, "pair" if big else "single")
        v = self.t[k]; v[0] += int(fl); v[1] += int(el); v[2] += 1
        self.ops[(self.phase, name)] += 1
        return out

def main():
    path, dest = sys.argv[1], sys.argv[2]
    repo = Path(os.environ["RFD4_REPO"])
    cfg, model = L.build_model(repo, torch.float32)
    batch, meta, _rec, _loss = L.load_batch(path)
    depth = int(os.environ.get("DEPTH", meta["depth"]))
    K = int(batch["padding_mask"].shape[0])
    take = lambda d: {k: take(v) for k, v in d.items()} if isinstance(d, dict) else (d[0:1] if torch.is_tensor(d) and d.dim() and d.shape[0] == K else d)
    nparam = sum(p.numel() for p in model.parameters())
    m = Count()
    with m:
        m.phase = "cand_k"
        with torch.no_grad():
            model.call_nn(dict(batch), n_recycle=depth)
        m.phase = "win_fwd"
        out = model.call_nn(take(dict(batch)), n_recycle=depth)
        loss = sum((v.float() ** 2).mean() for v in tree_flatten(out)[0] if torch.is_tensor(v) and v.requires_grad)
        m.phase = "win_bwd"
        loss.backward()
    shapes = [[list(p.shape), n] for n, p in model.named_parameters() if p.grad is not None]
    print("trained params (grad not None)", sum(p.numel() for p in model.parameters() if p.grad is not None))
    rows = [{"phase": k[0], "class": k[1], "size": k[2], "flops": v[0], "elems": v[1], "calls": v[2]} for k, v in sorted(m.t.items())]
    json.dump({"batch": path, "depth": depth, "K": K, "params": nparam, "rows": rows,
               "param_shapes": shapes, "top_ops": [[a, b, c] for (a, b), c in m.ops.most_common(80)]},
              open(dest, "w"), indent=1)
    print("params", nparam, "depth", depth, "K", K)
    for r in rows: print(r)

if __name__ == "__main__":
    main()
