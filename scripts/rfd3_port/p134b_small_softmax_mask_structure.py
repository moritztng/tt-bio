#!/usr/bin/env python3
"""p134b -- X5 part two: the structure of p134's error, and a negative control for the knob.

p134 measured the shape of the small softmax kernel's bf16 error and it is not a rounding
artefact: 98.4 % of elements differ, **56.25 % are exactly 0.0**, every one of the 256 tiles is
affected, and maxabs is exactly 0.97265625 at four seeds and three shapes. The `H4` column
matched for all 23 face permutations, which means it was measuring `max(ref) - ~0` rather than a
permutation, so it names nothing. Two questions are left and both are cheap:

  E  **negative control for `extra_defines`.** p134's hoisted-`typecast_tile_init` arm produced a
     byte-identical output to the shipped one. That is only evidence if a define CAN move the
     output, so compile with `P134_TYPECAST_SKIP`, which drops the SFPU conversion entirely and
     must land back on the packer's own rounding -- `perf/p74/softmax_generic.json`'s
     0.0009765625 at this shape. If it does not move, `extra_defines` never reached the compiler
     and p134's arm C is void (`negative-control-must-break-what-check-reads`).
  F  **the zero mask.** Is the same (row, col) set zero inside every tile? Which faces? Are the
     non-zero values a permutation of `ref`'s (a layout bug) or different numbers (arithmetic)?

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rfd3-fusion-programme-p5 \
      /home/moritz/tt-bio/env/bin/python3 -u \
      scripts/rfd3_port/p134b_small_softmax_mask_structure.py perf/p134/mask_structure.json

PROVISIONAL-ON-PC-CARD0.
"""
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
SHAPE = (1, 16, 128, 128)
PACKER_BASELINE = 0.0009765625     # perf/p74/softmax_generic.json, this shape, no SFPU typecast


def to_t(x):
    return ttnn.to_torch(x).float()


def run(dev, x, dtype, extra_defines=()):
    out = ttnn.empty(list(x.shape), dtype, ttnn.TILE_LAYOUT, x.device(), x.memory_config())
    softmax_generic.softmax_into(dev, x, out, extra_defines=extra_defines)
    t = to_t(out)
    ttnn.deallocate(out)
    return t


def tiles(t, W):
    flat = t.reshape(-1, W)
    R, C = flat.shape
    return flat.reshape(R // TILE, TILE, C // TILE, TILE).permute(0, 2, 1, 3).contiguous()


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p134/mask_structure.json")
    dev = get_device()
    W = SHAPE[-1]
    g = torch.Generator().manual_seed(42)
    h = torch.randn(SHAPE, generator=g, dtype=torch.float32) * 4.0
    x = ttnn.from_torch(h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    ref = ttnn.softmax(x, dim=-1)
    ref_bf = ttnn.typecast(ref, ttnn.bfloat16)
    refb = to_t(ref_bf)

    got = run(dev, x, ttnn.bfloat16)
    neg = run(dev, x, ttnn.bfloat16, extra_defines=(("P134_TYPECAST_SKIP", "1"),))
    hoist = run(dev, x, ttnn.bfloat16, extra_defines=(("TYPECAST_INIT_HOISTED", "1"),))

    rec = {"shape": list(SHAPE), "provisional_on": "pc-card0",
           "card": int(os.environ.get("TT_VISIBLE_DEVICES", "-1"))}

    # --- E: the negative control -------------------------------------------------------------
    rec["E_negctrl"] = {
        "shipped_maxabs": float((got - refb).abs().max()),
        "typecast_skipped_maxabs": float((neg - refb).abs().max()),
        "typecast_skipped_equals_packer_baseline":
            abs(float((neg - refb).abs().max()) - PACKER_BASELINE) < 1e-9,
        "negctrl_moved_the_output": not bool(torch.equal(neg, got)),
        "hoisted_maxabs": float((hoist - refb).abs().max()),
        "hoisted_moved_the_output": not bool(torch.equal(hoist, got)),
    }

    # --- F: the zero mask and the value multiset ----------------------------------------------
    G, R = tiles(got, W), tiles(refb, W)                      # (Ht, Wt, 32, 32)
    Ht, Wt = G.shape[0], G.shape[1]
    Z = (G == 0)                                              # zero mask per tile
    z0 = Z[0, 0]
    same_mask_everywhere = bool((Z == z0).all())
    per_face = [int(z0[r0:r0 + 16, c0:c0 + 16].sum())
                for r0 in (0, 16) for c0 in (0, 16)]
    rows_all_zero = [int(r) for r in torch.nonzero(z0.all(dim=1)).flatten()]
    cols_all_zero = [int(c) for c in torch.nonzero(z0.all(dim=0)).flatten()]
    # Is `got` a rearrangement of `ref` inside each tile?
    gs, _ = torch.sort(G.reshape(Ht, Wt, -1), dim=-1)
    rs, _ = torch.sort(R.reshape(Ht, Wt, -1), dim=-1)
    tile_multiset_equal = int((gs == rs).all(dim=-1).sum())
    # ... and per row of a tile, since the recip is broadcast down columns
    gsr, _ = torch.sort(G, dim=-1)
    rsr, _ = torch.sort(R, dim=-1)
    row_multiset_equal = int((gsr == rsr).all(dim=-1).sum())

    rec["F_mask"] = {
        "zeros_per_tile": int(z0.sum()), "tile_elements": TILE * TILE,
        "frac_zero": round(float(z0.float().mean()), 6),
        "same_zero_mask_in_every_tile": same_mask_everywhere,
        "n_tiles": Ht * Wt,
        "zeros_per_face_of_tile0": per_face,
        "rows_entirely_zero": rows_all_zero,
        "cols_entirely_zero": cols_all_zero,
        "zeros_per_row_of_tile0": [int(v) for v in z0.sum(dim=1)],
        "zeros_per_col_of_tile0": [int(v) for v in z0.sum(dim=0)],
        "mask_tile0": ["".join("0" if v else "#" for v in row) for row in z0.tolist()],
        "tiles_whose_value_multiset_matches_ref": tile_multiset_equal,
        "tile_rows_whose_value_multiset_matches_ref": row_multiset_equal,
        "tile_rows_total": Ht * Wt * TILE,
        "got_max": float(got.max()), "got_min": float(got.min()),
        "refb_max": float(refb.max()), "refb_min": float(refb.min()),
        "n_got_gt_1": int((got > 1.0).sum()),
    }

    e, f = rec["E_negctrl"], rec["F_mask"]
    print("[p134b] E  shipped %.9g | typecast-skipped %.9g (packer baseline %s, moved %s) | "
          "hoisted %.9g (moved %s)"
          % (e["shipped_maxabs"], e["typecast_skipped_maxabs"],
             e["typecast_skipped_equals_packer_baseline"], e["negctrl_moved_the_output"],
             e["hoisted_maxabs"], e["hoisted_moved_the_output"]))
    print("[p134b] F  zeros %d/%d per tile (%.4f), same mask in every tile: %s"
          % (f["zeros_per_tile"], f["tile_elements"], f["frac_zero"],
             f["same_zero_mask_in_every_tile"]))
    print("[p134b] F  zeros per face (tl,tr,bl,br) %s | rows all-zero %s | cols all-zero %s"
          % (f["zeros_per_face_of_tile0"], f["rows_entirely_zero"], f["cols_entirely_zero"]))
    print("[p134b] F  zeros per row of tile0 %s" % f["zeros_per_row_of_tile0"])
    print("[p134b] F  zeros per col of tile0 %s" % f["zeros_per_col_of_tile0"])
    print("[p134b] F  value multiset matches ref: %d/%d tiles, %d/%d tile-rows"
          % (f["tiles_whose_value_multiset_matches_ref"], f["n_tiles"],
             f["tile_rows_whose_value_multiset_matches_ref"], f["tile_rows_total"]))
    print("[p134b] F  got [%.6g, %.6g]  refb [%.6g, %.6g]  got>1: %d"
          % (f["got_min"], f["got_max"], f["refb_min"], f["refb_max"], f["n_got_gt_1"]))
    print("[p134b] F  zero mask of tile 0 ('0' = written as zero, '#' = non-zero):")
    for row in f["mask_tile0"]:
        print("           " + row)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print("[p134b] wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
