#!/usr/bin/env python3
"""Which tt-bio call sites issue the ttnn calls of one checkpointed Evoformer block's backward.

Count only (no timing): every `ttnn.<verb>` call made during `autograd.backward`, keyed by verb and
the innermost tt_bio frame that issued it, and the output's shape and dtype. One block, n=288, the
fold's masks, under `bindcraft2.fast_round()`, as the round runs it. Phase "fwd" is the untaped
forward, "bwd" the backward including its taped recompute.
"""
import collections, json, pathlib, sys, traceback
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VERBS = ["add", "subtract", "multiply", "mul", "div", "divide", "add_", "multiply_", "subtract_",
         "where", "typecast", "sum", "mean", "matmul", "linear", "permute", "transpose", "reshape",
         "concat", "slice", "clone", "neg", "exp", "sigmoid", "rsqrt", "pad", "to_layout",
         "experimental", "generic_op", "layer_norm", "softmax", "reciprocal", "square", "sqrt",
         "relu", "gt", "ge", "lt", "fill", "zeros_like", "ones_like", "full_like", "copy", "tilize",
         "untilize", "squeeze", "unsqueeze", "repeat", "expand", "max", "multiply_add", "addcmul",
         "minimal_matmul", "nlp_concat_heads", "nlp_create_qkv_heads", "sigmoid_bw", "mul_bw"]


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import bindcraft2
    from tt_bio.af2 import af2_pair_masks
    tr = bindcraft2._Trunk(pathlib.Path("/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"))
    ag, T = tr.ag, tr.taped
    n, nreal, depth = 288, 261, 2
    torch.manual_seed(0)
    m0, z0 = torch.randn(depth, n, 256) * 0.5, torch.randn(n, n, 128) * 0.5
    seq = torch.zeros(n); seq[:nreal] = 1
    msa_mask = tr.up(seq.expand(depth, n).contiguous())
    pm = af2_pair_masks(seq[:, None] * seq[None, :], tr.device)
    blk = tr.model.device_evoformer[0]
    phase = ["off"]
    counts = collections.Counter()

    def site():
        for fr in reversed(traceback.extract_stack()[:-2]):
            if "/tt_bio/" in fr.filename and "taped_ttnn.py" not in fr.filename[-15:] or \
               ("/tt_bio/" in fr.filename and fr.name not in ("call", "wrapped", "_run")):
                return f"{fr.filename.split('/tt_bio/')[-1]}:{fr.lineno}:{fr.name}"
        return "?"

    def wrap(name, fn):
        def w(*a, **k):
            out = fn(*a, **k)
            if phase[0] != "off":
                shp = dt = ""
                o = out[0] if isinstance(out, (list, tuple)) and out else out
                try:
                    shp, dt = str(list(o.shape)), str(o.dtype).split(".")[-1]
                except Exception:
                    pass
                counts[(phase[0], name, site(), shp, dt)] += 1
            return out
        return w

    for v in VERBS:
        f = getattr(ttnn, v, None)
        if callable(f) and not isinstance(f, type(ttnn)):
            setattr(ttnn, v, wrap(v, f))
    with bindcraft2.fast_round():
        for rep in range(2):
            phase[0] = "fwd" if rep else "off"
            ml, zl = tr.leaf(m0), tr.leaf(z0)
            with T.tape():
                m, z = ag.checkpoint(lambda x, y: blk(x, y, msa_mask, *pm), ml, zl)
            tr.sync()
            seeds = [tr.seed(torch.randn(m0.shape) * 1e-3, m), tr.seed(torch.randn(z0.shape) * 1e-3, z)]
            phase[0] = "bwd" if rep else "off"
            ag.backward([m, z], seeds)
            tr.sync()
            phase[0] = "off"
            ag.release_pins()
    rows = sorted(counts.items(), key=lambda kv: -kv[1])
    tot = collections.Counter()
    for (ph, v, s, shp, dt), c in rows:
        tot[(ph, v)] += c
    print(json.dumps({f"{p}:{v}": c for (p, v), c in tot.most_common()}))
    for (ph, v, s, shp, dt), c in rows:
        print(f"{ph}\t{c}\t{v}\t{s}\t{shp}\t{dt}")


if __name__ == "__main__":
    main()
