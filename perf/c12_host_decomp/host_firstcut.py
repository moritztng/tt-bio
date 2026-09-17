#!/usr/bin/env python3
"""First cut at the host line items, with no card, from two committed on-card measurements.

The row's own two-clock capture cannot run: qb2's board `...410D` has both chips stuck (see
CARD2_WEDGE.md). But two independent measurements already in the repo bracket the answer from
opposite ends, and combining them is arithmetic on committed numbers rather than a new instrument.

  * `perf/b2x_host_residual/residual_512_qb2c2.json` -- a CLOSED host tree over all of
    `predict_one` from one 512 aa fold on **qb2 card 2**, the same host, card type, fixture and
    protocol this row uses (p300c, 11x10, cdk2x2_512, 35-row MSA, 3 recycles, 200 steps,
    loadavg 2.34, torch 8 threads). 20 leaves summing to 23.54853 s against a 23.5521 s fold:
    unattributed 0.0035 s, 0.015 %. Its A/A floor was 0.16 % over two plain folds of the same
    process.
  * `c12-profiled-fold`'s `composed.json` -- the device term measured in situ at a held,
    during-sampled 1350 MHz: 13.2090 s of 14.8810 s, so **1.6489-1.6720 s** is the ceiling on all
    exposed host time in today's fold. That is what this row closes against.

Why the two do not simply add up, and why that is handled rather than ignored: the census fold is
**23.5521 s**, not 14.8810 s, because it predates three ports that have since landed default-on
(`TT_BIO_DEVICE_CONDITIONING`, `_ZINIT`, `_CONFIDENCE`). Those ports moved specific named rows of
that tree from host to device. So each leaf is classified explicitly, and only the leaves that are
still host today are summed. The classification is the argument; it is stated per leaf below and
each call is justified against either the port that moved it or a device unit that now owns it.

What this is NOT: it is not the row's table. These seconds come from a fold at a different total
wall and an unpinned clock, so no item here carries a clock arm and none of them can be called
CLOCK-IMMUNE by measurement. Host time should not scale with AICLK, which is exactly the
hypothesis the two-clock control exists to test, so assuming it here and then reporting it as a
result would be circular. This is a first cut that sizes the answer and says which items the
capture has to price, and it is labelled that way everywhere it appears.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS = HERE.parent / "b2x_host_residual/residual_512_qb2c2.json"
SCALING_128 = HERE.parent / "b2x_host_residual/preflight_128.json"

# Today's measured ceiling on all exposed host time, from c12-profiled-fold's composed.json.
REMAINDER = (1.6489, 1.6720)
FOLD_TODAY = 14.8810
REDUCIBLE_BAR = 0.8490          # what 12.5 s needs from host once every device lever lands

# Host inside the device units, measured in pass 2 off c12-profiled-fold's bare walls. It sits
# INSIDE the census tree's device leaves, so it is additional to the host leaves below and must
# not be double counted against them.
HOST_INSIDE_DEVICE_UNITS = 0.3372

# Per leaf: (class, why). Classes:
#   host    -- host Python then, host Python now. Counted.
#   moved   -- host Python then, on the device now. NOT counted: a port already deleted it.
#   device  -- the leaf's exclusive time is a device call's. NOT counted.
CLASS = {
    "predict_step/trunk":
        ("device", "TrunkModule, the resident trunk. 12.93 s of device time."),
    "predict_step/sampler/denoiser/denoise_device":
        ("device", "DiffusionModule, 200 calls. Pass 2 priced its host share separately at "
                   "0.1782 s off bare walls, which is inside HOST_INSIDE_DEVICE_UNITS."),
    "predict_step/confidence/pairformer_conf":
        ("device", "PairformerModule on the confidence path."),
    "predict_step/diffusion_cond":
        ("moved", "DiffusionConditioning.forward's own body. TT_BIO_DEVICE_CONDITIONING is "
                  "default True on this tree and PairConditioningDevice owns this work; pass 2 "
                  "measured its surviving host body at 0.1093 s, inside "
                  "HOST_INSIDE_DEVICE_UNITS."),
    "predict_step/diffusion_cond/pairwise_cond":
        ("moved", "PairwiseConditioning, folded into PairConditioningDevice by the same port."),
    "predict_step/diffusion_cond/atom_encoder":
        ("moved", "the z_to_p_trans projection PairConditioningDevice is constructed with "
                  "(tt_bio/boltz2.py _tt_cond_module passes dc.atom_encoder.z_to_p_trans)."),
    "predict_step/confidence":
        ("moved", "ConfidenceModule.forward's own body. TT_BIO_DEVICE_CONFIDENCE is default True "
                  "and c12-profiled-fold bounds ConfidenceHeadsDevice at <= 0.0231 s."),
    "predict_step/confidence/rel_pos":
        ("moved", "RelativePositionEncoder on the confidence path; RelPosGather is a device unit "
                  "in c12-profiled-fold's book, 2 calls, 0.0280 s of device time."),
    "predict_step/rel_pos":
        ("moved", "the trunk-path RelativePositionEncoder, same device unit."),

    "predict_step":
        ("host", "Boltz2.predict_step's own exclusive body. The largest surviving host row and "
                 "the one whose composition is unknown: the program-cache clear and the "
                 "reset_static_cache module walk both live in Boltz2.forward under it."),
    "prepare":
        ("host", "parse + MSA resolve + tokenize + featurize. Called ONCE per fold, in "
                 "tt_bio/worker.py:764, BEFORE predict_step, so it is outside the recycle loop."),
    "predict_step/sampler":
        ("host", "AtomDiffusion.sample's own loop body, the per-step arithmetic between the 200 "
                 "denoiser calls."),
    "predict_step/sampler/align":
        ("host", "weighted_rigid_align, 200 calls of host torch."),
    "write_result":
        ("host", "CIF write + metrics, inside the timed region "
                 "(model_meta.timed_region names it: featurize + fold + CIF write)."),
    "predict_step/input_embedder":
        ("host", "InputEmbedder.forward's own body, host torch with no device implementation."),
    "predict_step/input_embedder/atom_encoder":
        ("host", "AtomEncoder on the input-embedder path, host torch."),
    "predict_step/sampler/randaug":
        ("host", "compute_random_augmentation, 200 calls of host torch."),
    "predict_step/sampler/denoiser":
        ("host", "preconditioned_network_forward's own wrapper body, 200 calls."),
    "to_batch":
        ("host", "unsqueeze + move. 0.12 ms, a row only so the table has no glue remainder."),
    "predict_step/sampler/digest":
        ("host", "_write_sample_digest, not enabled on this protocol."),
}

# Reducibility, argued per item rather than assigned. `frac` is the fraction of the item's own
# seconds this row would claim as reducible WITHOUT moving work onto the device, because moving
# host torch to the card converts host seconds into device seconds instead of deleting them and
# the campaign's device book is already counted at full value against the same 12.5 s target.
REDUCIBLE = {
    "write_result": (1.00, "irreducible WORK, reducible COST. The CIF write sits inside the "
                           "timed region and nothing downstream in the fold consumes it, so it "
                           "can come off the critical path in full."),
    "prepare": (0.00, "real work that runs once. The brief's candidate was feature prep "
                      "recomputed per recycle; that is refuted at source (one call, before "
                      "predict_step, outside the recycle loop), so there is no invariant to "
                      "hoist. Any win here needs a faster featuriser, which is not free and is "
                      "not priced, so this row claims none of it until it is."),
    "predict_step": (0.00, "unknown composition and the row will not guess it. It holds the "
                           "program-cache clear, whose real cost is the program REBUILD and is "
                           "pre-registered separately at 0.0 s in a 0.0-2.3 s band, and the "
                           "reset_static_cache walk at 0.02 s. Both need the capture."),
    "predict_step/sampler": (0.00, "per-step host arithmetic. Reducible only onto the device, "
                                   "and 200 tiny programs run into the per-program launch floor, "
                                   "so it is not obviously a win there."),
    "predict_step/sampler/align": (0.00, "same: 200 calls of host torch, device-only route."),
    "predict_step/sampler/randaug": (0.00, "same."),
    "predict_step/sampler/denoiser": (0.00, "wrapper body, 18.5 ms over 200 calls."),
    "predict_step/input_embedder": (0.00, "host torch, no device implementation. Device-only "
                                          "route."),
    "predict_step/input_embedder/atom_encoder": (0.00, "same."),
    "to_batch": (0.00, "0.12 ms."),
    "predict_step/sampler/digest": (0.00, "zero."),
}


def scaling(tree_512):
    """Does each host item grow with the problem, or is it fixed overhead? Two on-card sizes.

    `preflight_128.json` is a second CLOSED tree from the SAME card (qb2 card 2, p300c):
    4.9736 s fold, unattributed 0.0004 s. 512/128 tokens is a **4.0x** ratio, against the
    1.72x of the 298-to-512 pair the row's `SCALING:` field actually asks for, so this is a
    bigger lever on the same question and not a substitute for that field.

    Three things keep this honest:

      * **Only leaves present in BOTH trees are compared.** The preflight instrument carried a
        smaller patch set: it has no `diffusion_cond`, `rel_pos` or `input_embedder` rows. A
        missing row is not a zero and is reported as absent.
      * **Two points cannot fit an exponent** and none is fitted. The ratio is reported and the
        only claim made from it is the qualitative one: grows with the problem, or does not.
      * **The 128 leg ran at loadavg 5.91 against the 512 leg's 2.34.** Host rows inflate under
        load, so the 128 seconds are if anything too high, which makes every ratio here a LOWER
        bound for the items that scale. It is the conservative direction for the conclusion.
    """
    try:
        small = json.loads(SCALING_128.read_text())
    except OSError as e:
        return {"error": repr(e)}
    t128, a128, e128 = small["attrib"]["tree"], small["attrib"], small["env"]
    ratio_tokens = 512.0 / float(e128["size"])
    rows = []
    for path, v in sorted(tree_512.items(), key=lambda kv: -kv[1]["excl_s"]):
        klass, _ = CLASS[path]
        if klass != "host":
            continue
        if path not in t128:
            rows.append({"item": path, "s_512": v["excl_s"], "s_128": None,
                         "ratio": None, "note": "absent from the preflight patch set"})
            continue
        a, b = v["excl_s"], t128[path]["excl_s"]
        r = (a / b) if b > 1e-9 else None
        rows.append({"item": path, "s_512": a, "s_128": b,
                     "ratio": round(r, 3) if r else None,
                     "verdict": ("too small to call" if a < 0.001 else
                                 "grows with the problem" if r and r >= 2.0 else
                                 "flat in size" if r and r <= 1.5 else
                                 "weakly size-dependent")})
    grows = sum(r["s_512"] for r in rows if r.get("verdict") == "grows with the problem")
    flat = sum(r["s_512"] for r in rows
               if r.get("verdict") in ("flat in size", "weakly size-dependent"))
    return {
        "compared_against": str(SCALING_128.name),
        "small_fold_s": a128["fold_wall_s"], "small_unattributed_s": a128["unattributed_s"],
        "small_size_tokens": e128["size"], "token_ratio": ratio_tokens,
        "small_loadavg": a128["loadavg"], "large_loadavg": None,
        "caveat": "ratios are LOWER bounds: the 128 leg ran at loadavg 5.91 against 2.34, and "
                  "host rows inflate under load. Only leaves in both trees are compared, and no "
                  "exponent is fitted from two points.",
        "host_s_512_that_grows": round(grows, 5),
        "host_s_512_that_is_flat": round(flat, 5),
        "reading": "the host seconds at 512 aa are dominated by items that grow with the "
                   "problem, and the per-step sampler glue is near flat in size, so it does not "
                   "get worse at the sizes the campaign cares about",
        "rows": rows,
    }


def main() -> int:
    c = json.loads(CENSUS.read_text())
    tree, env, attrib = c["attrib"]["tree"], c["env"], c["attrib"]
    missing = set(tree) ^ set(CLASS)
    if missing:
        print(f"census tree and classification disagree on: {sorted(missing)}", file=sys.stderr)
        return 2

    rows = []
    for path, v in sorted(tree.items(), key=lambda kv: -kv[1]["excl_s"]):
        klass, why = CLASS[path]
        frac, arg = REDUCIBLE.get(path, (0.0, "not a host row"))
        rows.append({"item": path, "calls": v["calls"], "excl_s": v["excl_s"], "class": klass,
                     "why": why,
                     "reducible_s": round(v["excl_s"] * frac, 5) if klass == "host" else 0.0,
                     "reducible_frac": frac if klass == "host" else None,
                     "reducible_argument": arg if klass == "host" else None})

    host = sum(r["excl_s"] for r in rows if r["class"] == "host")
    moved = sum(r["excl_s"] for r in rows if r["class"] == "moved")
    device = sum(r["excl_s"] for r in rows if r["class"] == "device")
    reducible = sum(r["reducible_s"] for r in rows)
    named = host + HOST_INSIDE_DEVICE_UNITS

    out = {
        "what": "FIRST CUT, no card. Not the row's table and carries no clock arm.",
        "provenance": {
            "census": str(CENSUS.relative_to(CENSUS.parents[2])),
            "census_fold_s": attrib["fold_wall_s"],
            "census_unattributed_s": attrib["unattributed_s"],
            "census_card": f"{env['card']} {env['card_type']}",
            "census_commit": env["commit"],
            "census_loadavg": attrib["loadavg"],
            "census_torch_threads": env["torch_threads"],
            "census_timed_region": env["timed_region"],
            "remainder_source": "c12-profiled-fold composed.json, device 13.2090 s of 14.8810 s "
                                "in situ at a held during-sampled 1350 MHz",
        },
        "census_split_s": {"host_today": round(host, 5), "moved_to_device_since": round(moved, 5),
                           "device": round(device, 5),
                           "sum": round(host + moved + device, 5)},
        "closure_first_cut": {
            "host_leaves_s": round(host, 5),
            "host_inside_device_units_s": HOST_INSIDE_DEVICE_UNITS,
            "named_total_s": round(named, 5),
            "remainder_s": list(REMAINDER),
            "named_pct_of_remainder": [round(100 * named / REMAINDER[1], 1),
                                       round(100 * named / REMAINDER[0], 1)],
            "unnamed_s": [round(REMAINDER[0] - named, 4), round(REMAINDER[1] - named, 4)],
        },
        "reducible_first_cut": {
            "reducible_s": round(reducible, 5),
            "bar_s": REDUCIBLE_BAR,
            "fraction_of_bar": round(reducible / REDUCIBLE_BAR, 3),
            "shortfall_s": round(REDUCIBLE_BAR - reducible, 4),
            "generous_ceiling_s": round(reducible + 0.5 * (
                tree["prepare"]["excl_s"] + tree["predict_step"]["excl_s"]), 5),
            "generous_argument": "even crediting half of prepare and half of predict_step's "
                                 "unknown exclusive body, neither of which is priced, the total "
                                 "stays below the bar. Everything past that point is host torch "
                                 "whose only route is onto the device, which converts host "
                                 "seconds into device seconds rather than deleting them.",
            "fold_if_all_named_host_deleted_s": round(FOLD_TODAY - named, 4),
        },
        "rows": rows,
        "scaling_first_cut": scaling(tree),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
