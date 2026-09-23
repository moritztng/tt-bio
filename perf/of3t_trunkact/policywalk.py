#!/usr/bin/env python3
"""Upstream 0.4.3's 48-block pairformer stack at padded N=384 in THREE dtype policies at once,
walked block by block in ONE process, scoring our device dump at every block.

`refwalk.py` runs the first two. This adds the third, which is the one that decides whether the
trunk's residual has an op in it at all: `torch.autocast` keeps a whole CLASS of operations in
float32 -- layer norms, softmax, reductions -- while a bf16 device kernel does them in bf16. A
port measured against an autocast reference therefore pays for the policy difference at every
one of those sites, and A27 already records that autocast and full-cast differ by 4.08x
somewhere in this campaign. If our arm lands at the full-cast level, the residual IS the policy
and no single op carries it.

Four arms, one function by construction (one boundary, one checkpoint, one interpreter):

  f64        every parameter and every activation float64, checkpoint upcast once at load, no
             cast anywhere on the path (A27). This is the reference.
  bf16auto   float32 parameters under `torch.autocast('cpu', bfloat16)` -- upstream's own
             training recipe, and the FLOOR our arm is measured against. A27 exists because a
             full-cast bf16 arm and an autocast one differ by 4.08x, so the policy is named.
  bf16full   every parameter and every activation bfloat16, autocast OFF, so every op including
             the normalisations and the softmax runs at bf16 -- the policy a bf16 device kernel
             actually implements. NOT a claim about what upstream runs; it is the other end of
             the policy axis, measured so the middle can be read.
  ours       `perf/of3t_trunkact/devwalk.py`'s per-block dump, read off disk.

What it reports, and why it is a DIFFERENCE and not a share: `mass_weighted_rel_l2`-style
shares move when their denominator moves, and this campaign has been misled by that twice
(`a-share-moves-when-its-denominator-collapses`). So the per-block quantity here is the
ABSOLUTE error norm `|| ours_k - ref64_k ||` and the reported step is its first difference
across k. Relative columns are published beside it, never instead of it.

PAD vs REAL is split on both tracks, because the gradient this row exists to explain is
computed over the padded tensor: `dW = sum_t g_t xhat_t` runs over all 384 token positions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"
PRE = "pairformer_stack.blocks."


def build(sd, n, dtype):
    """Upstream's own PairFormerBlock, dimensions read off the checkpoint, strict load."""
    from openfold3.core.model.latent.pairformer import PairFormerBlock
    p0 = f"{PRE}0."
    c_s = sd[p0 + "attn_pair_bias.layer_norm_a.weight"].shape[0]
    c_z = sd[p0 + "pair_stack.tri_mul_in.layer_norm_in.weight"].shape[0]
    nh_bias = sd[p0 + "attn_pair_bias.linear_z.weight"].shape[0]
    nh_pair = sd[p0 + "pair_stack.tri_att_start.linear_z.weight"].shape[0]
    dims = dict(c_s=c_s, c_z=c_z, c_hidden_pair_bias=c_s // nh_bias, no_heads_pair_bias=nh_bias,
                c_hidden_mul=sd[p0 + "pair_stack.tri_mul_in.linear_a_p.weight"].shape[0],
                c_hidden_pair_att=sd[p0 + "pair_stack.tri_att_start.mha.linear_q.weight"].shape[0]
                // nh_pair,
                no_heads_pair=nh_pair, transition_type="swiglu",
                transition_n=sd[p0 + "pair_stack.pair_transition.swiglu.linear_a.weight"].shape[0]
                // c_z,
                pair_dropout=0.25, fuse_projection_weights=False, inf=1e9)
    mods, missing, unexpected, n_t = [], [], [], 0
    for i in range(n):
        sub = {k[len(f"{PRE}{i}."):]: v.to(dtype) for k, v in sd.items()
               if k.startswith(f"{PRE}{i}.")}
        m = PairFormerBlock(**dims).to(dtype)
        miss, unex = m.load_state_dict(sub, strict=True)
        missing += [f"{i}.{x}" for x in miss]
        unexpected += [f"{i}.{x}" for x in unex]
        n_t += len(sub)
        mods.append(m.eval())
    return mods, dims, {"tensors_loaded": n_t, "missing": missing, "unexpected": unexpected}


def cmp(ours, ref, keep):
    """One comparison, absolute first. `keep` is a boolean mask broadcastable over ref."""
    o = ours.to(torch.float64)
    r = ref.to(torch.float64)
    if keep is not None:
        o = o * keep
        r = r * keep
    d = o - r
    dn, rn, on = float(d.norm()), float(r.norm()), float(o.norm())
    return {"abs_err": dn, "ref_norm": rn, "ours_norm": on,
            "rel_l2": (dn / rn) if rn > 0 else None,
            "norm_ratio": (on / rn) if rn > 0 else None,
            "cos": (float((o * r).sum()) / (on * rn)) if on > 0 and rn > 0 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--ours", action="append", required=True, metavar="NAME=DIR",
                    help="one or more devwalk dumps, scored against the same reference walk")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ref-dump", default="", help="optional dir to keep ref64 per-block state")
    ap.add_argument("--cap-last", default="", help="captured cotangent, for the pad control")
    a = ap.parse_args()
    t0 = time.perf_counter()

    sys.path.insert(0, a.tree)
    import openfold3
    if not openfold3.__file__.startswith(a.tree):
        raise SystemExit(f"wrong tree on sys.path: {openfold3.__file__} not under {a.tree}")
    torch.set_num_threads(a.threads)  # reference walk: upstream's tree only, no tt_bio import
    torch.manual_seed(0)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m64, dims, load64 = build(sd, a.blocks, torch.float64)
    m32, _, load32 = build(sd, a.blocks, torch.float32)
    m16, _, load16 = build(sd, a.blocks, torch.bfloat16)

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    sm_raw, pm_raw = b["single_mask"], b["pair_mask"]
    N = int(b["z_in"].shape[1])
    real = (sm_raw.reshape(-1) > 0)
    n_real = int(real.sum())
    # pad/real selectors. The single track is [1, N, c_s] so the split is on token rows; the
    # pair track is [1, N, N, c_z] so it is on cells, and a cell is REAL only if both of its
    # tokens are.
    s_real = real.reshape(1, N, 1).to(torch.float64)
    z_real = (real.reshape(1, N, 1) & real.reshape(1, 1, N)).reshape(1, N, N, 1).to(torch.float64)
    sel = {"padded": {"s": None, "z": None},
           "real": {"s": s_real, "z": z_real},
           "pad": {"s": 1.0 - s_real, "z": 1.0 - z_real}}

    s64 = b["s_in"].to(torch.float64).contiguous()
    z64 = b["z_in"].to(torch.float64).contiguous()
    sm64, pm64 = sm_raw.to(torch.float64), pm_raw.to(torch.float64)
    s32 = b["s_in"].to(torch.float32).contiguous()
    z32 = b["z_in"].to(torch.float32).contiguous()
    sm32, pm32 = sm_raw.to(torch.float32), pm_raw.to(torch.float32)
    s16 = b["s_in"].to(torch.bfloat16).contiguous()
    z16 = b["z_in"].to(torch.bfloat16).contiguous()
    sm16, pm16 = sm_raw.to(torch.bfloat16), pm_raw.to(torch.bfloat16)

    # CONTROL, and it is the linchpin of this row's pad exclusion: dW = sum_t g_t xhat_t runs
    # over every token, so the pad activations reach the parameter gradient only if the
    # cotangent is non-zero there. Read it off the capture rather than inferred from the
    # pad-zero identity being bit-exact.
    COT = {"note": "the captured cotangent entering block 47, split pad vs real"}
    if a.cap_last:
        c = torch.load(a.cap_last, map_location="cpu", weights_only=False)["cot"]
        cs, cz = c[0].to(torch.float64), c[1].to(torch.float64)
        COT["cap_last"] = a.cap_last
        COT["cot_s_norm_total"] = float(cs.norm())
        COT["cot_s_norm_real"] = float((cs * s_real).norm())
        COT["cot_s_norm_pad"] = float((cs * (1.0 - s_real)).norm())
        COT["cot_z_norm_total"] = float(cz.norm())
        COT["cot_z_norm_real"] = float((cz * z_real).norm())
        COT["cot_z_norm_pad"] = float((cz * (1.0 - z_real)).norm())
        COT["cot_pad_is_exactly_zero"] = bool(COT["cot_s_norm_pad"] == 0.0
                                              and COT["cot_z_norm_pad"] == 0.0)
        COT["n_nonzero_pad_elements_s"] = int((cs * (1.0 - s_real) != 0).sum())
        COT["n_nonzero_pad_elements_z"] = int((cz * (1.0 - z_real) != 0).sum())
        del cs, cz, c
        print(json.dumps(COT), flush=True)

    if a.ref_dump:
        os.makedirs(a.ref_dump, exist_ok=True)

    arms = {}
    for spec in a.ours:
        nm, _, d = spec.partition("=")
        if not d:
            nm, d = "ours", spec
        arms[nm] = d

    rows = []
    with torch.no_grad():
        for k in range(a.blocks):
            tb = time.perf_counter()
            with torch.autocast("cpu", enabled=False):
                s64, z64 = m64[k](s64, z64, sm64, pm64)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                s32, z32 = m32[k](s32, z32, sm32, pm32)
            s32 = s32.to(torch.float32)
            z32 = z32.to(torch.float32)
            with torch.autocast("cpu", enabled=False):
                s16, z16 = m16[k](s16, z16, sm16, pm16)
            row = {"block": k, "seconds": time.perf_counter() - tb,
                   "ref64": {"s_norm": float(s64.norm()), "z_norm": float(z64.norm())}}
            for scope in ("padded", "real", "pad"):
                row[scope] = {"bf16_s": cmp(s32, s64, sel[scope]["s"]),
                              "bf16_z": cmp(z32, z64, sel[scope]["z"]),
                              "full16_s": cmp(s16, s64, sel[scope]["s"]),
                              "full16_z": cmp(z16, z64, sel[scope]["z"])}
            for nm, d in arms.items():
                o = torch.load(os.path.join(d, f"blk{k:02d}.pt"))
                for scope in ("padded", "real", "pad"):
                    row[scope][f"{nm}_s"] = cmp(o["s"], s64, sel[scope]["s"])
                    row[scope][f"{nm}_z"] = cmp(o["z"], z64, sel[scope]["z"])
                del o
            rows.append(row)
            if a.ref_dump:
                torch.save({"s": s64.clone(), "z": z64.clone()},
                           os.path.join(a.ref_dump, f"blk{k:02d}.pt"))
            nm0 = next(iter(arms))
            print(json.dumps({"block": k, "arm": nm0,
                              "s_abs": row["padded"][f"{nm0}_s"]["abs_err"],
                              "z_abs": row["padded"][f"{nm0}_z"]["abs_err"],
                              "s_pad_share_sq": round(
                                  (row["pad"][f"{nm0}_s"]["abs_err"] ** 2)
                                  / max(row["padded"][f"{nm0}_s"]["abs_err"] ** 2, 1e-300), 6),
                              "z_pad_share_sq": round(
                                  (row["pad"][f"{nm0}_z"]["abs_err"] ** 2)
                                  / max(row["padded"][f"{nm0}_z"]["abs_err"] ** 2, 1e-300), 6),
                              "full16_s_real": row["real"]["full16_s"]["abs_err"],
                              "bf16_s_real": row["real"]["bf16_s"]["abs_err"],
                              "ours_s_real": row["real"][f"{nm0}_s"]["abs_err"],
                              "full16_z_real": row["real"]["full16_z"]["abs_err"],
                              "ours_z_real": row["real"][f"{nm0}_z"]["abs_err"],
                              "sec": round(row["seconds"], 1)}), flush=True)

    # first differences of the ABSOLUTE error, which is the step statistic
    def diffs(key, leaf, scope="padded"):
        v = [r[scope][f"{key}_{leaf}"]["abs_err"] for r in rows]
        return [None] + [v[i] - v[i - 1] for i in range(1, len(v))]

    rep = {"what": __doc__.strip().splitlines()[0],
           "host": os.uname().nodename, "tree": a.tree,
           "openfold3_file": openfold3.__file__,
           "blocks": a.blocks, "tokens": N, "real_tokens": n_real,
           "dims": dims, "load_f64": load64, "load_f32": load32, "load_bf16": load16,
           "boundary": a.boundary, "arms": arms,
           "dtype_policy": {
               "f64": "every parameter and every activation float64; checkpoint upcast once at "
                      "load; no cast on the path",
               "bf16auto": "float32 parameters under torch.autocast('cpu', bfloat16), "
                           "upstream's own training recipe",
               "bf16full": "every parameter and every activation bfloat16, autocast OFF"},
           "final_ref64": {"s_norm": float(s64.norm()), "z_norm": float(z64.norm())},
           "final_bf16": {"s_norm": float(s32.to(torch.float64).norm()),
                          "z_norm": float(z32.to(torch.float64).norm())},
           "final_full16": {"s_norm": float(s16.to(torch.float64).norm()),
                            "z_norm": float(z16.to(torch.float64).norm())},
           "cotangent_pad_control": COT,
           "per_block": rows,
           "abs_err_first_difference": {
               f"{nm}_{leaf}{'' if sc == 'padded' else '_' + sc}": diffs(nm, leaf, sc)
               for nm in list(arms) + ["bf16", "full16"] for leaf in ("s", "z")
               for sc in ("padded", "pad", "real")},
           "seconds": time.perf_counter() - t0}
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("per_block", "abs_err_first_difference", "dims",
                                   "load_f64", "load_f32")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
