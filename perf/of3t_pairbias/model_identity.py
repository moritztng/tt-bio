"""Does splitting `scale_pair_bias` move a model that did not ask for the split?

`tri_att_scale_pair_bias=None` makes `tri_scale` fall back to `scale_pair_bias`, so a caller
that does not name the new argument hands `TriangleAttention` exactly the value it handed
before. That is an argument. This is the measurement: build each model's pair stack from its
OWN checkpoint with its OWN production flags, run one fixed forward, sha256 the output bytes.
Run it here and from a detached `origin/wk/of3t` checkout on the same card; the two digests
agree or the split moved something.

`--tri 1` is the control the comparison needs. A digest check that cannot see the flag move
proves nothing about it not moving, so every model is also run with the new argument FORCED
and must come out with a DIFFERENT digest. On the `origin/wk/of3t` side that run raises
TypeError, which is itself the proof the two checkouts are not the same code.

Production flags, copied from the real construction sites and named per entry so an entry
cannot outlive its module:

    boltz2, boltzgen  tenstorrent.py PairformerFactory._create_module -- all defaults
    protenix          protenix.py:2463 (v2 trunk)  gated_move, accurate_softmax
    af2               NOT a Pairformer: af2.py:330 builds TriangleAttention directly, so the
                      split cannot reach it by construction. Measured anyway, via AF2PairBlock.

    python3 perf/of3t_pairbias/model_identity.py --model protenix [--tri 1]
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))

OUT = "perf/of3t_pairbias/model_identity.json"
MODELS = ("boltz2", "boltzgen", "protenix", "af2")


def prod_flags(model):
    from tt_bio.tenstorrent import accurate_softmax_site
    if model == "protenix":
        return dict(gated_move=True,
                    accurate_softmax=accurate_softmax_site("protenix-v2.trunk", default=True))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--tri", default="none", choices=("none", "0", "1"),
                    help="force tri_att_scale_pair_bias; 'none' is the production call")
    a = ap.parse_args()

    import torch
    import ttnn
    from perf.of3t_leaves import leaf_census as LC
    from tt_bio.tenstorrent import Pairformer, get_device

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    forced = {} if a.tri == "none" else {"tri_att_scale_pair_bias": bool(int(a.tri))}
    if a.model == "af2":
        if forced:
            print("af2 builds TriangleAttention directly, there is no PairformerLayer to force")
            return 3
        mod, order, shapes, note = LC.build("af2", a.blocks, ckc)
    else:
        flat = LC._truncate(LC.REMAPS[a.model](), a.blocks)
        nb = 1 + max(int(k.split(".")[1]) for k in flat if k.startswith("layers."))
        c_z = flat["layers.0.tri_mul_out.norm_in.weight"].shape[0]
        c_s = flat["layers.0.pre_norm_s.weight"].shape[0]
        n_heads = flat["layers.0.tri_att_start.linear.weight"].shape[0]
        head_dim = flat["layers.0.tri_att_start.mha.linear_q.weight"].shape[0] // n_heads
        apb_nh = flat["layers.0.attention.proj_z.1.weight"].shape[0]
        dims = (head_dim, n_heads, c_s // apb_nh, apb_nh)
        kw = dict(prod_flags(a.model), **forced)
        mod = Pairformer(nb, *dims, True, flat, ckc, **kw)
        order, shapes = ("s", "z"), {"c_z": c_z, "c_s": c_s, "dims": dims}
        note = f"{nb}-block Pairformer, dims {dims}, flags {sorted(kw)}"

    n = a.tokens
    rng = torch.Generator().manual_seed(7)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    raw = {}
    if shapes["c_s"] is not None:
        raw["s"] = ft(torch.randn(1, n, shapes["c_s"], generator=rng) * 0.05)
    raw["z"] = ft(torch.randn(1, n, n, shapes["c_z"], generator=rng) * 0.05)
    out = mod(*[raw[k] for k in order])
    if not isinstance(out, tuple):
        out = (out,)
    h, parts = hashlib.sha256(), []
    for t in out:
        if t is None:
            h.update(b"None")
            parts.append(None)
            continue
        x = torch.Tensor(ttnn.to_torch(t)).float().contiguous()
        h.update(x.numpy().tobytes())
        parts.append([list(x.shape), float(x.double().norm())])
    row = dict(model=a.model, tri=a.tri, note=note, blocks=a.blocks, tokens=n,
               sha256=h.hexdigest(), outputs=parts)
    print(f"{a.model:9s} tri={a.tri:4s} {row['sha256']}  {note}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    rows = [r for r in rows if (r["model"], r.get("tri")) != (a.model, a.tri)] + [row]
    json.dump(rows, open(OUT, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
