#!/usr/bin/env python3
"""P5 / C5 — hoist the loop-invariant template z projection, measured and parity-checked.

Also re-measures both stage walls on THIS card, because T5's are pc card 0, and settles whether the
2D `to_layout` defect this leg found in isolation shows up in the live `trunk_msa` stage.

`Trunk._template` (`tt_bio/protenix.py:2021-2026`) computes `zn = LN(z3)` once, then re-evaluates
`self._lin(zn, "template_embedder.linear_no_bias_z.weight")` inside `for t in range(nt)`. Operand
and weight are both loop-invariant, so nt-1 of nt calls are identical work discarded.

The hoisted variant is a LOCAL function in this probe. Nothing under `tt_bio/` is modified.
"""
from __future__ import annotations

import argparse, json, statistics as st, sys, time
from pathlib import Path

import torch, ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import (get_device, CORE_GRID_MAIN, OuterProductMean,   # noqa: E402
                                PairWeightedAveraging, PairformerLayer, Transition)
from tt_bio import protenix_weights as PW                                        # noqa: E402
from tt_bio.protenix import Trunk                                                # noqa: E402


def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}


def build_trunk(sd, ckc, c_z):
    t = Trunk.__new__(Trunk)
    t._w, t.compute_kernel_config, t.dev, t.C_Z = sd, ckc, get_device(), c_z
    t._wc, t._msa_update_first = {}, False
    n_tri_heads = c_z // Trunk.TRI_HEAD_DIM

    def sub(pp):
        return {k[len(pp):]: v for k, v in sd.items() if k.startswith(pp)}

    t.TPL = [PairformerLayer(32, 2, None, None, False,
                             PW.remap_msa_pair_stack(sub(f"template_embedder.pairformer_stack.blocks.{b}.")),
                             ckc) for b in range(2)]
    t.MSA = []
    for i in range(4):
        P = f"msa_module.blocks.{i}."
        opm = OuterProductMean(PW.remap_outer_product_mean(sub(P + "outer_product_mean_msa.")), ckc)
        pl = PairformerLayer(Trunk.TRI_HEAD_DIM, n_tri_heads, None, None, False,
                             PW.remap_msa_pair_stack(sub(P + "pair_stack.")), ckc)
        has = any(k.startswith(P + "msa_stack.") for k in sd)
        pwa = tm = None
        if has:
            pwa = PairWeightedAveraging(8, 8, PW.remap_pair_weighted_averaging(
                sub(P + "msa_stack.msa_pair_weighted_averaging.")), ckc)
            tm = Transition(PW.remap_transition(sub(P + "msa_stack.transition_m.")), ckc)
        t.MSA.append((opm, pwa, tm, pl))
    t.PF = None
    return t


def template_hoisted(t, z3, tpl_a, N, nt):
    """`Trunk._template` with the one loop-invariant projection lifted out of the loop.

    Byte-for-byte the production body otherwise. This is the whole candidate.
    """
    zn = t._ln(z3, "template_embedder.layernorm_z.weight", "template_embedder.layernorm_z.bias")
    zp = t._lin(zn, "template_embedder.linear_no_bias_z.weight")          # <-- hoisted
    u = None
    for i in range(nt):
        v = ttnn.add(tpl_a[i], zp)
        for pl in t.TPL:
            v = pl(None, v)[1]
        v = t._ln(v, "template_embedder.layernorm_v.weight", "template_embedder.layernorm_v.bias")
        u = v if u is None else ttnn.add(u, v)
    u = ttnn.multiply(u, 1.0 / (1e-7 + nt))
    return t._lin(ttnn.relu(u), "template_embedder.linear_no_bias_u.weight")


def wall(dev, fn, warm, reps):
    for _ in range(warm):
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        out.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
    return st.median(out), [round(x * 1e3, 2) for x in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path.home() / ".boltz" / "protenix-v2.pt"))
    ap.add_argument("--n", type=int, default=298)
    ap.add_argument("--depth", type=int, default=35)
    ap.add_argument("--nt", type=int, default=4)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = load_sd(args.ckpt)
    c_z = sd["template_embedder.linear_no_bias_z.weight"].shape[1]
    N, D, nt = args.n, args.depth, args.nt
    print(f"N={N} depth={D} nt={nt} c_z={c_z} grid={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}", flush=True)
    t = build_trunk(sd, ckc, c_z)
    torch.manual_seed(0)
    z3 = ttnn.from_torch(torch.randn(1, N, N, c_z) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16)
    m_feat = ttnn.from_torch(torch.randn(1, D, N, sd["msa_module.linear_no_bias_m.weight"].shape[0]) * 0.1,
                             layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    tpl_a = [t._lin(t._up(torch.randn(1, N, N, 108) * 0.1),
                    "template_embedder.linear_no_bias_a.weight") for _ in range(nt)]

    res = {"device": {"host": "qb2", "card": 1, "ttnn": "0.68.0"}, "n": N, "nt": nt, "depth": D}

    # ---- C5: baseline vs hoisted, stage wall + parity -----------------------------------------
    base_s, base_raw = wall(dev, lambda: t._template(z3, tpl_a, N, nt), args.warm, args.reps)
    hoi_s, hoi_raw = wall(dev, lambda: template_hoisted(t, z3, tpl_a, N, nt), args.warm, args.reps)
    ob = ttnn.to_torch(t._template(z3, tpl_a, N, nt))
    oh = ttnn.to_torch(template_hoisted(t, z3, tpl_a, N, nt))
    eq = bool(torch.equal(ob, oh))
    md = float((ob.float() - oh.float()).abs().max())
    res["c5"] = {
        "template_baseline_ms": round(base_s * 1e3, 3), "template_baseline_raw_ms": base_raw,
        "template_hoisted_ms": round(hoi_s * 1e3, 3), "template_hoisted_raw_ms": hoi_raw,
        "delta_ms_per_stage": round((base_s - hoi_s) * 1e3, 3),
        "delta_ms_per_fold_x10": round((base_s - hoi_s) * 1e4, 1),
        "baseline_ms_per_fold_x10": round(base_s * 1e4, 1),
        "pct_of_stage": round(100 * (base_s - hoi_s) / base_s, 2),
        "torch_equal": eq, "max_abs_diff": md,
        "calls_removed_per_fold": (nt - 1) * 10}
    print("C5: " + json.dumps(res["c5"]), flush=True)

    # ---- the z projection on its own, and its core ladder -------------------------------------
    zn = t._ln(z3, "template_embedder.layernorm_z.weight", "template_embedder.layernorm_z.bias")
    lad = {}
    for gx, gy in ((4, 4), (8, 4), (11, 8), (11, 10)):
        g = ttnn.CoreGrid(x=gx, y=gy)
        w = t._w_tt("template_embedder.linear_no_bias_z.weight")
        o = []
        for _ in range(3):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(3):
                ttnn.deallocate(ttnn.linear(zn, w, compute_kernel_config=ckc, core_grid=g,
                                            dtype=ttnn.bfloat16))
            ttnn.synchronize_device(dev)
            o.append((time.perf_counter() - t0) / 3)
        lad[f"{gx}x{gy}"] = round(st.median(o) * 1e6, 1)
        print(f"  z_proj {gx}x{gy}: {lad[f'{gx}x{gy}']} us", flush=True)
    res["c5_core_ladder_us"] = lad

    # ---- trunk_msa stage wall on this card: does the 2D to_layout defect show up live? ---------
    def msa_once():
        out = t._msa(z3, m_feat)
        if isinstance(out, list):
            for o2 in out[1:]:
                ttnn.deallocate(o2)
            return out[0]
        return out
    msa_s, msa_raw = wall(dev, msa_once, 1, 3)
    res["trunk_msa"] = {"stage_ms": round(msa_s * 1e3, 3), "ms_per_fold_x10": round(msa_s * 1e4, 1),
                        "raw_ms": msa_raw,
                        "t5_pc_ms_per_fold": 1979.3}
    print("msa: " + json.dumps(res["trunk_msa"]), flush=True)

    (args.out / "p5_c5_hoist.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
