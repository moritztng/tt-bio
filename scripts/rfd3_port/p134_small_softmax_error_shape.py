#!/usr/bin/env python3
"""p134 -- X5: what the small softmax kernel's bf16 error actually LOOKS like, before any fix.

Three mechanism guesses have failed (p2: `typecast_tile_init` in two positions inside the acquire
block, and the `cb_out0` reserve ordering). State doc §15.6 forbids a fourth guess and asks for the
error's shape instead. This script measures it and scores the one untried placement in the same
process; it does not propose a mechanism of its own.

What the artifacts already say, and this run re-checks on pc card 0 (they were taken on qb2):
`perf/p74/softmax_generic.json` has S2 maxabs 0.00098-0.00195 everywhere (packer rounding, one
bf16 ULP). `perf/p74/softmax_generic_sfpu.json` and `..._cand1.json` have the large kernel exact
and the small kernel off by **exactly 0.97265625 at four different shapes** -- Wt 4/16/22/107,
blk 4/4/2/1, four different random inputs. A maxabs that is constant across shape, block size and
data is not a rounding artefact.

Arms, all in one process and one device context:

  gate  S1 (fp32 out) must still be `torch.equal`. If it is not, nothing below means anything.
  A     the error's shape, unprejudiced: per-tile maxabs over the (Ht, Wt) grid, the count of
        exactly-equal elements, the value histogram of `got` where it differs and of `ref` there,
        and the argmax decomposed to (tile_h, tile_w, face, row-in-face, col-in-face).
  B     the four hypotheses §15.6 named, scored against the DEVICE `ref` rather than a host
        softmax: unnormalised `exp(x-max)`; fp32 bits read as two bf16 halves; `ref` truncated
        rather than round-to-nearest (the packer's own rounding); `ref` with the four 16x16 tile
        faces permuted, all 23 non-identity permutations.
  C     the untried placement: `typecast_tile_init` once, before `mul_bcast_cols_init_short`,
        outside every acquire block -- which is what the bit-exact large kernel's `apply_recip`
        does. Compiled in the same process through `softmax_into(extra_defines=...)`.
  D     seeds: is 0.97265625 a constant of the code or of the data?

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rfd3-fusion-programme-p5 \
      /home/moritz/tt-bio/env/bin/python3 -u scripts/rfd3_port/p134_small_softmax_error_shape.py \
          perf/p134/small_softmax_error_shape.json

PROVISIONAL-ON-PC-CARD0: this is a softmax kernel with no matmul in it, so card 0's known
matmul defect (`pc-card0-512aa-fold-nondeterminism`) has no site here, but the label stands.
"""
import itertools
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.getcwd())
import ttnn                                                              # noqa: E402
from tt_bio import softmax_generic                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

TILE = 32
HOISTED = (("TYPECAST_INIT_HOISTED", "1"),)
# [1,16,128,128] is the smallest failing instance: Wt=4, blk=4, units = NC*Ht = 64 <= the core
# grid, so every core runs NCHt=1 and the normalise loop runs exactly one iteration. There is no
# next iteration for an init to clobber and no cross-`ncht` state, which is what makes it the
# right place to look.
SMALL = (1, 16, 128, 128)


def to_t(x):
    return ttnn.to_torch(x).float()


def run(dev, x, out_dtype, extra_defines=()):
    out = ttnn.empty(list(x.shape), out_dtype, ttnn.TILE_LAYOUT, x.device(), x.memory_config())
    softmax_generic.softmax_into(dev, x, out, extra_defines=extra_defines)
    got = to_t(out)
    ttnn.deallocate(out)
    return got


def tile_view(t, W):
    """(Ht, Wt, 32, 32) over the folded [H, W] view `plan` works in."""
    flat = t.reshape(-1, W)
    R, C = flat.shape
    return flat.reshape(R // TILE, TILE, C // TILE, TILE).permute(0, 2, 1, 3).contiguous()


def face_perms(refb, W):
    """`refb` with the four 16x16 faces of every tile permuted. Yields (perm, tensor)."""
    T = tile_view(refb, W)
    Ht, Wt = T.shape[0], T.shape[1]
    F = (T.reshape(Ht, Wt, 2, 16, 2, 16).permute(0, 1, 2, 4, 3, 5).contiguous()
          .reshape(Ht, Wt, 4, 16, 16))
    for p in itertools.permutations(range(4)):
        if p == (0, 1, 2, 3):
            continue
        G = (F[:, :, list(p)].reshape(Ht, Wt, 2, 2, 16, 16).permute(0, 1, 2, 4, 3, 5)
             .reshape(Ht, Wt, TILE, TILE))
        yield p, G.permute(0, 2, 1, 3).reshape(Ht * TILE, Wt * TILE)


def maxabs(a, b):
    return float((a - b).abs().max())


def hypotheses(x_t, ref32, refb, W):
    """The four §15.6 hypotheses, host-computed from the DEVICE fp32 reference."""
    out = {}
    un = torch.exp(x_t - x_t.amax(-1, keepdim=True))
    out["H1_unnormalised_exp_x_minus_max"] = un.bfloat16().float().reshape(-1, W)
    bits = ref32.contiguous().view(torch.int32)
    out["H2a_fp32_high_half_as_bf16"] = ((bits >> 16 << 16).view(torch.float32)).reshape(-1, W)
    out["H2b_fp32_low_half_as_bf16"] = (((bits & 0xFFFF) << 16).view(torch.float32)).reshape(-1, W)
    out["H3_packer_truncate_not_rne"] = ((bits & -65536).view(torch.float32)).reshape(-1, W)
    return out


def error_shape(got, refb, W):
    """Arm A. Everything about the error that does not assume a mechanism."""
    g, r = got.reshape(-1, W), refb.reshape(-1, W)
    d = (g - r).abs()
    n = d.numel()
    neq = int((g != r).sum())
    Tg, Tr = tile_view(g, W), tile_view(r, W)
    per_tile = (Tg - Tr).abs().amax(dim=(-1, -2))                 # (Ht, Wt)
    bad_tiles = int((per_tile > 0).sum())
    idx = int(d.argmax())
    row, col = idx // W, idx % W
    diff = g[g != r]
    dref = r[g != r]
    gv, gc = torch.unique(diff, return_counts=True)
    rv, rc = torch.unique(dref, return_counts=True)
    top = lambda v, c, k=8: [[float(v[i]), int(c[i])]
                             for i in torch.argsort(c, descending=True)[:k]]
    return {
        "n_elements": n,
        "n_exactly_equal": n - neq,
        "frac_exactly_equal": round((n - neq) / n, 6),
        "maxabs": maxabs(g, r),
        "tile_grid": [int(per_tile.shape[0]), int(per_tile.shape[1])],
        "n_tiles_with_any_error": bad_tiles,
        "n_tiles": int(per_tile.numel()),
        "per_tile_maxabs_unique": [[float(v), int(c)] for v, c in
                                   zip(*torch.unique(per_tile, return_counts=True))][:12],
        "per_tile_maxabs_first_rows": per_tile[:4].tolist(),
        "argmax": {"flat_row": row, "col": col,
                   "tile_h": row // TILE, "tile_w": col // TILE,
                   "face": (row % TILE) // 16 * 2 + (col % TILE) // 16,
                   "row_in_face": (row % TILE) % 16, "col_in_face": (col % TILE) % 16,
                   "got": float(g[row, col]), "ref": float(r[row, col])},
        "got_value_histogram_where_wrong": top(gv, gc),
        "ref_value_histogram_where_wrong": top(rv, rc),
        "n_distinct_got_values_where_wrong": int(gv.numel()),
        "got_is_a_single_constant_where_wrong": bool(gv.numel() == 1),
    }


def one_shape(dev, shape, seed, do_full):
    W = shape[-1]
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(shape, generator=g, dtype=torch.float32) * 4.0
    x = ttnn.from_torch(h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    ref = ttnn.softmax(x, dim=-1)
    ref_bf = ttnn.typecast(ref, ttnn.bfloat16)
    ref32, refb = to_t(ref), to_t(ref_bf)
    p = softmax_generic.plan(x, ref_bf, (dev.compute_with_storage_grid_size().x,
                                         dev.compute_with_storage_grid_size().y), True, True)
    rec = {"shape": list(shape), "seed": seed, "Wt": p["Wt"], "Ht": p["Ht"],
           "block_size": p["block_size"], "use_large": p["use_large"],
           "units": p["units"], "target_cores": p["target"], "per1": p["per1"],
           "ncht_per_core": p["per1"]}

    got32 = run(dev, x, ttnn.float32)
    rec["S1_fp32_equal"] = bool(torch.equal(got32, ref32))
    rec["S1_maxabs"] = maxabs(got32, ref32)

    shipped = run(dev, x, ttnn.bfloat16)
    rec["S2_shipped_equal"] = bool(torch.equal(shipped, refb))
    rec["S2_shipped_maxabs"] = maxabs(shipped, refb)

    hoist = run(dev, x, ttnn.bfloat16, extra_defines=HOISTED)
    rec["S2_hoisted_equal"] = bool(torch.equal(hoist, refb))
    rec["S2_hoisted_maxabs"] = maxabs(hoist, refb)
    rec["hoisted_differs_from_shipped"] = not bool(torch.equal(hoist, shipped))

    if do_full:
        rec["arm_A_error_shape"] = error_shape(shipped, refb, W)
        hyp = {k: maxabs(shipped.reshape(-1, W), v)
               for k, v in hypotheses(to_t(x), ref32, refb, W).items()}
        perm = {"".join(map(str, pm)): maxabs(shipped.reshape(-1, W), t)
                for pm, t in face_perms(refb, W)}
        best = min(perm.items(), key=lambda kv: kv[1])
        hyp["H4_tile_face_permuted_best"] = best[1]
        rec["arm_B_hypotheses_maxabs"] = hyp
        rec["arm_B_H4_best_perm"] = best[0]
        rec["arm_B_H4_all_perms"] = perm
        rec["arm_B_any_hypothesis_reproduces_S2"] = any(
            abs(v - rec["S2_shipped_maxabs"]) < 1e-9 for v in hyp.values())

    for t in (x, ref, ref_bf):
        ttnn.deallocate(t)
    return rec


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                       else "perf/p134/small_softmax_error_shape.json")
    dev = get_device()
    rows = []
    rows.append(one_shape(dev, SMALL, 42, True))
    # Arm D: three more seeds at the same shape. If S2's maxabs does not move with the data, the
    # wrong value is not a function of the input.
    for s in (7, 11, 1234):
        rows.append(one_shape(dev, SMALL, s, False))
    # The two other small rungs that failed on qb2, and the large kernel as a control: the large
    # one was already bit-exact and this pass does not touch it.
    for shape in ((1, 16, 512, 512), (1, 16, 685, 704), (1, 4, 4540, 4544)):
        rows.append(one_shape(dev, shape, 42, False))

    for r in rows:
        print("[p134] %-22s seed %-5d Wt=%-4d blk=%-2d large=%-5s ncht/core=%-3d  "
              "S1 %-5s  S2 shipped %-5s (%.9g)  S2 hoisted %-5s (%.9g)"
              % (str(tuple(r["shape"])), r["seed"], r["Wt"], r["block_size"], r["use_large"],
                 r["ncht_per_core"], r["S1_fp32_equal"], r["S2_shipped_equal"],
                 r["S2_shipped_maxabs"], r["S2_hoisted_equal"], r["S2_hoisted_maxabs"]),
              flush=True)

    a = rows[0].get("arm_A_error_shape", {})
    if a:
        print("\n[p134] arm A, at %s seed 42:" % (tuple(rows[0]["shape"]),))
        print("   exactly equal        %d / %d  (%.4f)"
              % (a["n_exactly_equal"], a["n_elements"], a["frac_exactly_equal"]))
        print("   tiles with any error %d / %d over the %s grid"
              % (a["n_tiles_with_any_error"], a["n_tiles"], a["tile_grid"]))
        print("   argmax               %s" % a["argmax"])
        print("   got where wrong      %d distinct, single constant: %s"
              % (a["n_distinct_got_values_where_wrong"],
                 a["got_is_a_single_constant_where_wrong"]))
        print("   got histogram        %s" % a["got_value_histogram_where_wrong"])
        print("   ref histogram        %s" % a["ref_value_histogram_where_wrong"])
        print("   per-tile maxabs      %s" % a["per_tile_maxabs_unique"])
        print("\n[p134] arm B, hypotheses vs S2 maxabs %.9g:" % rows[0]["S2_shipped_maxabs"])
        for k, v in rows[0]["arm_B_hypotheses_maxabs"].items():
            print("   %-34s %.9g %s" % (k, v,
                  "<-- REPRODUCES S2" if abs(v - rows[0]["S2_shipped_maxabs"]) < 1e-9 else ""))
        print("   any hypothesis reproduces S2: %s"
              % rows[0]["arm_B_any_hypothesis_reproduces_S2"])

    small = [r for r in rows if not r["use_large"]]
    rec = {"rows": rows, "provisional_on": "pc-card0",
           "card": int(os.environ.get("TT_VISIBLE_DEVICES", "-1")),
           "s1_all_equal": all(r["S1_fp32_equal"] for r in rows),
           "s2_shipped_all_equal": all(r["S2_shipped_equal"] for r in rows),
           "s2_hoisted_all_equal": all(r["S2_hoisted_equal"] for r in rows),
           "s2_hoisted_fixes_small": all(r["S2_hoisted_equal"] for r in small),
           "s2_shipped_maxabs_constant_over_seeds":
               len({r["S2_shipped_maxabs"] for r in rows if not r["use_large"]}) == 1}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print("\n[p134] S1 all equal %s | S2 shipped all equal %s | S2 hoisted fixes every small "
          "shape %s | shipped maxabs constant across seeds/shapes %s"
          % (rec["s1_all_equal"], rec["s2_shipped_all_equal"], rec["s2_hoisted_fixes_small"],
             rec["s2_shipped_maxabs_constant_over_seeds"]))
    print("[p134] wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
