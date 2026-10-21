"""RFD4's attention family one op at a time, at the trunk's shape, against the levers bet 6 weighs.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/e2e/attn_bench.py --weights <bundle> --out <file.json> \
        [--n 128] [--reps 50] [--block 1]

Each op runs `--reps` times back to back from the same precomputed inputs and is synced once at the
end, so the time per op is what the card spends on it with dispatch overlapped, the way the trunk
issues it. Inputs are seeded at the released widths with every token real except the last 8, so the
masks are live.

Rows:
* `ops`: every op of one `AttnADALN` call, in the base form (capped SDPA chunk).
* `sdpa`: the shipped program config (q_chunk from `_capped_sdpa_chunk_size`, 12 work units at
  n 128) against the grid rule's pick (`work` = batch x heads), with a bit-compare.
* `pair_bias`: the shipped ttnn chain (cast, rms0, linear, permute, mask add, slices) against
  rms0 + `dit_pair_bias`, both graded against float64 on the host from the same bf16 z.
* `qkvg`: four [768 -> 768] linears against one [768 -> 3072] linear plus the slices the QK norm
  then needs.
* `whole`: one `AttnADALN` call with each SDPA config, and one bias group by each path.

AICLK and the host load are sampled for the life of the run (`perf/clocksample.py`).
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
import ttnn  # noqa: E402


def rel_rms(a, ref):
    a, ref = a.double(), ref.double()
    return float((a - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--block", type=int, default=1, help="a block that opens a two-block bias group")
    a = ap.parse_args()

    from tt_bio import dit_pair_bias as DPB
    from tt_bio import tenstorrent as T
    from tt_bio.rfd4 import network as N
    from tt_bio.rfd4 import trunk as TR

    dev = T.get_device()
    state, config = N.load_checkpoint(a.weights)
    cfg = config["nn"]
    trunk = TR.Trunk(T.WeightScope({k: v.float() for k, v in N._sub(state, "nn.").items()}),
                     TR.compute_kernel_config(), nlayers=cfg["nlayers"], n_heads=cfg["nheads"],
                     every_n=cfg["update_pair_repr_every_n"], residual_dtype=ttnn.float32)
    ckc = trunk.compute_kernel_config
    n, tok, pair, dc = a.n, cfg["token_dim"], cfg["pair_repr_dim"], cfg["dim_cond"]
    g = torch.Generator().manual_seed(0)
    mask = torch.ones(1, n)
    mask[:, -8:] = 0
    s0 = torch.randn(1, n, tok, generator=g) * mask[..., None]
    z0 = torch.randn(1, n, n, pair, generator=g) * (mask[:, :, None] * mask[:, None, :])[..., None]
    c0 = torch.randn(1, n, dc, generator=g)
    up = lambda t, dt: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    masks = TR.trunk_masks(mask, dev, residual_dtype=ttnn.float32)
    tmask, bias_mask = masks[0], masks[1]
    x = up(s0, ttnn.bfloat16)
    z = up(z0, ttnn.float32)
    cond = up(c0, ttnn.bfloat16)
    sync = lambda: ttnn.synchronize_device(dev)  # noqa: E731
    att = trunk.blocks[a.block].mhba
    mha = att.mha
    k_group = trunk._group_at[a.block]
    group = trunk.bias_groups[k_group]
    assert len(group) == 2, group
    H, hd = mha.n_heads, mha.head_dim

    def free(t):
        for u in (t if isinstance(t, (list, tuple)) else [t]):
            if u is not None and u.is_allocated():
                ttnn.deallocate(u)

    def timed(fn, reps=a.reps):
        free(fn())  # compile
        sync()
        t = time.perf_counter()
        outs = [fn() for _ in range(reps)]
        sync()
        us = 1e6 * (time.perf_counter() - t) / reps
        for o in outs:
            free(o)
        return round(us, 2)

    def best(fn, legs=3):
        return min(timed(fn) for _ in range(legs))

    rec = {"n": n, "heads": H, "head_dim": hd, "token_dim": tok, "pair": pair, "reps": a.reps,
           "tt_bio": str(Path(TR.__file__).resolve().parents[2])}
    with clocksample.during(period=0.5) as clk:
        load0 = os.getloadavg()[0]
        cn = TR._rms(cond, None, ckc)
        mask_scaled = ttnn.multiply(bias_mask, hd ** 0.5)
        # --- one AttnADALN call, op by op, shipped form
        ad, sc = att.adaln, att.scale_output
        xn = TR._rms(x, None, ckc)
        gamma = ad._lin(cn, ad.gamma_w, bias=ad.gamma_b, dtype=ad.dtype, activation="sigmoid")
        beta = ad._lin(cn, ad.beta_w, dtype=ad.dtype)
        xg = ttnn.multiply(xn, gamma)
        xgb = ttnn.add(xg, beta)
        h = ttnn.multiply(xgb, tmask)
        an = TR._rms(h, mha.node_norm_w, ckc)
        q = mha._lin(an, mha.q_w, bias=mha.q_b, dtype=mha.dtype)
        k = mha._lin(an, mha.k_w, bias=mha.k_b, dtype=mha.dtype)
        v = mha._lin(an, mha.v_w, bias=mha.v_b, dtype=mha.dtype)
        gt = mha._lin(an, mha.g_w, bias=mha.g_b, dtype=mha.dtype, activation="sigmoid")
        qn, kn = TR._rms(q, mha.q_norm_w, ckc), TR._rms(k, mha.k_norm_w, ckc)
        qkv = ttnn.unsqueeze(ttnn.concat([qn, kn, v], dim=-1), 1)
        heads = lambda: ttnn.experimental.nlp_create_qkv_heads(  # noqa: E731
            qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qh, kh, vh = heads()
        biases = TR.pair_biases(ttnn.typecast(z, ttnn.bfloat16), trunk.bias_ws[k_group][0], mask_scaled, ckc,
                                ttnn.bfloat16, n_out=2)
        bias = biases[0]
        pc = lambda work: T._sdpa_program_config_for_lengths(n, n, work, site="rfd4_trunk", d=hd)  # noqa: E731
        sdpa = lambda cfg_: T.fused_sdpa(qh, kh, vh, attn_mask=bias, scale=hd ** -0.5,  # noqa: E731
                                         compute_kernel_config=ckc, program_config=cfg_)
        o = sdpa(pc(0))
        om = mha._merge_heads(o)
        gated = ttnn.multiply(om, gt)
        out = mha._lin(gated, mha.out_w, bias=mha.out_b, dtype=mha.dtype)
        sg = sc._lin(cond, sc.w, bias=sc.b, dtype=sc.dtype, activation="sigmoid")
        so = ttnn.multiply(out, sg)
        ops = {
            "adaln.rms0": lambda: TR._rms(x, None, ckc),
            "adaln.gamma_lin_sigmoid": lambda: ad._lin(cn, ad.gamma_w, bias=ad.gamma_b, dtype=ad.dtype,
                                                       activation="sigmoid"),
            "adaln.beta_lin": lambda: ad._lin(cn, ad.beta_w, dtype=ad.dtype),
            "adaln.mul": lambda: ttnn.multiply(xn, gamma),
            "adaln.add": lambda: ttnn.add(xg, beta),
            "adaln.mask": lambda: ttnn.multiply(xgb, tmask),
            "mha.node_rms": lambda: TR._rms(h, mha.node_norm_w, ckc),
            "mha.q_lin": lambda: mha._lin(an, mha.q_w, bias=mha.q_b, dtype=mha.dtype),
            "mha.k_lin": lambda: mha._lin(an, mha.k_w, bias=mha.k_b, dtype=mha.dtype),
            "mha.v_lin": lambda: mha._lin(an, mha.v_w, bias=mha.v_b, dtype=mha.dtype),
            "mha.g_lin_sigmoid": lambda: mha._lin(an, mha.g_w, bias=mha.g_b, dtype=mha.dtype,
                                                  activation="sigmoid"),
            "mha.q_rms": lambda: TR._rms(q, mha.q_norm_w, ckc),
            "mha.k_rms": lambda: TR._rms(k, mha.k_norm_w, ckc),
            "mha.concat_qkv": lambda: ttnn.concat([qn, kn, v], dim=-1),
            "mha.split_heads": heads,
            "mha.sdpa": lambda: sdpa(pc(0)),
            "mha.merge_heads": lambda: mha._merge_heads(o),
            "mha.gate_mul": lambda: ttnn.multiply(om, gt),
            "mha.out_lin": lambda: mha._lin(gated, mha.out_w, bias=mha.out_b, dtype=mha.dtype),
            "scale.lin_sigmoid": lambda: sc._lin(cond, sc.w, bias=sc.b, dtype=sc.dtype, activation="sigmoid"),
            "scale.mul": lambda: ttnn.multiply(out, sg),
            "scale.mask": lambda: ttnn.multiply(so, tmask),
        }
        rec["ops_us"] = {name: best(fn) for name, fn in ops.items()}
        rec["ops_sum_us"] = round(sum(rec["ops_us"].values()), 1)

        # --- SDPA: shipped q_chunk against the grid rule's pick (q_chunk partitions query rows)
        ship, grid = pc(0), pc(int(qh.shape[0]) * H)
        ref_o, grid_o = ttnn.to_torch(sdpa(ship)), ttnn.to_torch(sdpa(grid))
        rec["sdpa"] = {"shipped": {"q_chunk": ship.q_chunk_size, "k_chunk": ship.k_chunk_size,
                                   "us": best(lambda: sdpa(ship))},
                       "grid": {"q_chunk": grid.q_chunk_size, "k_chunk": grid.k_chunk_size,
                                "us": best(lambda: sdpa(grid))},
                       "bit_identical": bool(torch.equal(ref_o, grid_o))}
        for qc in (32, 64, 128):
            cfg_ = T._sdpa_program_config(qc, ship.k_chunk_size)
            rec["sdpa"][f"q{qc}_k{ship.k_chunk_size}_us"] = best(lambda: sdpa(cfg_))

        # --- pair bias: shipped chain against rms0 + dit_pair_bias, both against float64
        zb = ttnn.typecast(z, ttnn.bfloat16)
        zn = TR._rms(zb, None, ckc)
        w2 = torch.cat([trunk.blocks[i].mhba.mha.bias_w_host for i in group], dim=1)  # [pair, 2H]
        w_tt, wt = trunk.bias_ws[k_group]
        chain_call = lambda: TR.pair_biases(zb, w_tt, mask_scaled, ckc, ttnn.bfloat16, n_out=2)  # noqa: E731
        rec["pair_bias"] = {"dpb_eligible": bool(DPB.eligible(zn, wt, mask_scaled)),
                            "dpb_l1_bytes": DPB._l1_bytes(pair // 32, 1, DPB._stream(pair // 32, 1)),
                            "dpb_streams_z": DPB._stream(pair // 32, 1)}
        chain_ops = {
            "cast_z_bf16": lambda: ttnn.typecast(z, ttnn.bfloat16),
            "rms0": lambda: TR._rms(zb, None, ckc),
            "linear": lambda: ttnn.linear(zn, w_tt, compute_kernel_config=ckc,
                                          dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN),
        }
        lin = chain_ops["linear"]()
        chain_ops["permute"] = lambda: ttnn.permute(lin, (0, 3, 1, 2))
        perm = chain_ops["permute"]()
        chain_ops["mask_add"] = lambda: ttnn.add(perm, mask_scaled)
        both = chain_ops["mask_add"]()
        chain_ops["slices"] = lambda: [ttnn.slice(both, (0, j * H, 0, 0), (1, (j + 1) * H, n, n)) for j in range(2)]
        chain_ops["dit_pair_bias"] = lambda: DPB.dit_pair_bias(zn, wt, mask_scaled, 1.0, n_out=2, h_valid=2 * H)
        rec["pair_bias"]["ops_us"] = {name: best(fn) for name, fn in chain_ops.items()}
        rec["pair_bias"]["whole_chain_us"] = best(chain_call)
        rec["pair_bias"]["whole_dpb_us"] = best(lambda: trunk._group_biases(k_group, zb, mask_scaled))
        # float64 reference from the same bf16 z
        zr = ttnn.to_torch(zb).double()
        znr = zr / (zr.pow(2).mean(-1, keepdim=True) + TR.EPS).sqrt()
        ref = (znr @ w2.double()).permute(0, 3, 1, 2) + (ttnn.to_torch(bias_mask).double() * hd ** 0.5)
        real = mask[0].bool()
        sel = lambda t: t[:, :, real][:, :, :, real]  # noqa: E731 -- graded on the real pairs only
        chain = torch.cat([ttnn.to_torch(t) for t in chain_call()], dim=1)
        dpb = torch.cat([ttnn.to_torch(t) for t in DPB.dit_pair_bias(zn, wt, mask_scaled, 1.0, n_out=2,
                                                                       h_valid=2 * H)], dim=1)
        rec["pair_bias"]["shape"] = list(dpb.shape)
        rec["pair_bias"]["rel_rms_vs_f64"] = {"chain": rel_rms(sel(chain), sel(ref)), "dpb": rel_rms(sel(dpb), sel(ref))}
        rec["pair_bias"]["masked_pairs_equal_mask"] = bool(
            torch.equal(dpb[:, :, ~real].float(), (ttnn.to_torch(mask_scaled).expand_as(dpb))[:, :, ~real].float()))
        rec["pair_bias"]["dpb_vs_chain_max_abs_real"] = float((sel(dpb) - sel(chain)).abs().max())

        # --- q/k/v/g: four linears against one plus the slices the packed-axis QK norm needs
        wq = torch.cat([mha.weights["to_qkv.weight"], mha.weights["to_g.weight"]]).float().t()
        bq = torch.cat([mha.weights["to_qkv.bias"], mha.weights["to_g.bias"]]).float()
        w4, b4 = up(wq, ttnn.bfloat16), up(bq[None], ttnn.bfloat16)
        big = mha._lin(an, w4, bias=b4, dtype=mha.dtype)
        inner = mha.inner
        rec["qkvg"] = {
            "four_linears_us": round(sum(rec["ops_us"][f"mha.{p}"] for p in ("q_lin", "k_lin", "v_lin",
                                                                            "g_lin_sigmoid")), 2),
            "one_linear_us": best(lambda: mha._lin(an, w4, bias=b4, dtype=mha.dtype)),
            "four_slices_us": best(lambda: [ttnn.slice(big, (0, 0, j * inner), (1, n, (j + 1) * inner))
                                            for j in range(4)]),
            "sigmoid_us": best(lambda: ttnn.sigmoid(v)),
        }
        rec["qkvg"]["fused_total_us"] = round(rec["qkvg"]["one_linear_us"] + rec["qkvg"]["four_slices_us"]
                                              + rec["qkvg"]["sigmoid_us"], 2)

        # --- whole calls
        attn = lambda: att(x, zb, cond, tmask, bias_mask, cn=cn, bias=bias)  # noqa: E731
        rec["whole_us"] = {"attn_adaln_grid_sdpa": best(attn)}
        orig = T._sdpa_program_config_for_lengths
        TR._sdpa_program_config_for_lengths = lambda q_len, k_len, work=0, **kw: orig(q_len, k_len, 0, **kw)
        rec["whole_us"]["attn_adaln_capped_sdpa"] = best(attn)
        TR._sdpa_program_config_for_lengths = orig
        rec["whole_us"]["group_biases_chain"] = rec["pair_bias"]["whole_chain_us"]
        rec["whole_us"]["group_biases_dpb"] = rec["pair_bias"]["whole_dpb_us"]
        rec["host_load_1m"] = [load0, os.getloadavg()[0]]
    rec["aiclk"] = clk.summary()
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
