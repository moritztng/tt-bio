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
import statistics
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
#   mixed   -- part of it moved and part of it did not, and the census cannot separate them
#              because it did not patch below this leaf. NOT an upper bound on today's value,
#              for a reason measured on another part: see UNNAMED_IS_THE_ANSWER below.
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
        ("mixed", "NOT all host today, and the first cut had this wrong. `predict_step` is a thin "
                  "wrapper that calls `Boltz2.forward` and copies dict entries, and the census "
                  "patched `predict_step` but NOT `forward`, so this leaf is `forward`'s own "
                  "body. At the census commit `_device_zinit()` was off, so that body BUILT "
                  "z_init on the host: z_init_1 + z_init_2 + token_bonds + contact_conditioning "
                  "and five adds over a [1,n,n,token_z] tensor that is 134 MB at 512 tokens, plus "
                  "a torch.zeros_like of the same. `_device_zinit()` is ON by default today "
                  "(tt_bio/boltz2.py:1198), so `z_init is None` and that whole path is skipped; "
                  "what it costs now is inside PairAssemblyDevice, whose host body pass 2 "
                  "measured at 0.0224 s. What survives in this leaf is the program-cache clear, "
                  "the reset_static_cache module walk, s_init, pair_mask, the gate evaluations "
                  "and the progress emissions. The census cannot split the two and neither can "
                  "this row without its own capture, so the leaf is an upper bound."),
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
    # This row credited this item at 1.00 for one pass and then audited it, because the one
    # number a row credits is the one it should check hardest. The credit does not survive.
    #
    # `model_meta.timed_region` is "predict_one (featurize + fold + CIF write)". The 14.8810 s
    # cell, the 12.5 s target and every fold second the campaign quotes are measured over a
    # region that ENDS WITH THE FILE ON DISK. Deferring the write past the timer does not make
    # the fold faster, it makes the timer stop earlier -- which is the campaign's own standing
    # prohibition on buying a speedup by doing less of the work, not a lever. For a single fold
    # measured to completion the wall until the CIF exists is unchanged by any amount of
    # deferral, because nothing follows it to overlap with.
    #
    # It IS worth 0.0570 s in the pipelined multi-target case, where target N's write overlaps
    # target N+1's fold. That is how JapanFold actually serves, so it is a real throughput win
    # and it is recorded as one below. It is not a latency win against the 12.5 s cell, and this
    # row's bar is the cell.
    "write_result": (0.00, "irreducible against the campaign's own timed region, which ends with "
                           "the file on disk. Deferring it stops the timer earlier rather than "
                           "finishing sooner. Worth 0.0570 s of THROUGHPUT in a pipelined "
                           "multi-target service, recorded separately, and nothing against the "
                           "12.5 s latency cell."),
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


def perturbation():
    """Does the instrument this table depends on change the fold it measures?

    The brief's warning is the right one: "a profiler that changes the fold time it is measuring
    is reporting its own overhead". Both census artifacts ran their arms IN ONE PROCESS on the
    same card, so the comparison is paired rather than across sessions:

      plain   no instrument at all
      attrib  the bracket region tree installed -- about 20 `perf_counter` pairs, and the
              instrument every number in this table comes from
      sample  the bracket tree PLUS an in-process stack sampler (128 aa only)

    What this does NOT answer is the field the row owes, which is the fold wall with `cProfile`
    and with `py-spy` attached at 512 aa. `cProfile` is deliberately the heaviest instrument in
    `decomp.py`'s arm list and nothing here measures it. So this is reported as what it is: the
    perturbation of the CHEAP instrument, which is the one the table's own seconds depend on.
    """
    out = {}
    for name, f, size in (("512", CENSUS, 512), ("128", SCALING_128, 128)):
        try:
            d = json.loads(f.read_text())
        except OSError:
            continue
        plain = [v["wall_s"] for v in d.get("plain", {}).values() if "wall_s" in v]
        if not plain:
            continue
        base = statistics.median(plain)   # mean of the two for an even count
        row = {"plain_folds_s": plain, "plain_median_s": base,
               "attrib_s": d["attrib"]["fold_wall_s"],
               "attrib_ratio": round(d["attrib"]["fold_wall_s"] / base, 4),
               "loadavg_plain": [v.get("loadavg") for v in d.get("plain", {}).values()],
               "loadavg_attrib": d["attrib"].get("loadavg")}
        if "sample" in d and d["sample"]:
            row["sample_s"] = d["sample"]["fold_wall_s"]
            row["sample_ratio_vs_plain"] = round(d["sample"]["fold_wall_s"] / base, 4)
            row["sample_ratio_vs_attrib"] = round(
                d["sample"]["fold_wall_s"] / d["attrib"]["fold_wall_s"], 4)
        out[name] = row

    if "512" in out:
        out["512"]["aa_floor_pct"] = 0.16
        out["512"]["reading"] = (
            "0.9990x against a 0.16 % A/A floor from the two plain folds of the same process, so "
            "the bracket instrument's perturbation at 512 aa is UNRESOLVABLE: it is smaller than "
            "the session's own noise. The table's seconds are the fold's, not the instrument's.")
    if "128" in out:
        out["128"]["reading"] = (
            "the instrumented folds come out FASTER than the plain one here, which is not a "
            "negative overhead. The plain fold ran first in the process and the box was at "
            "loadavg 6.25, so this is warm-up plus noise and the only honest statement is that "
            "the effect is below the noise at this size too. The stack sampler adds 1.0090x on "
            "top of the brackets, which is the one resolvable number in this block.")
    out["still_owed"] = ("the fold wall with cProfile and with py-spy attached at 512 aa. "
                         "cProfile is the heaviest arm in decomp.py and nothing committed "
                         "measures it, so PERTURBATION: stays unfilled.")
    return out


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
        if klass not in ("host", "mixed"):
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
    mixed = sum(r["excl_s"] for r in rows if r["class"] == "mixed")
    moved = sum(r["excl_s"] for r in rows if r["class"] == "moved")
    device = sum(r["excl_s"] for r in rows if r["class"] == "device")
    reducible = sum(r["reducible_s"] for r in rows)
    named_lo = host + HOST_INSIDE_DEVICE_UNITS
    named_hi = named_lo + mixed

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
        "census_split_s": {"host_today": round(host, 5),
                           "mixed_upper_bound": round(mixed, 5),
                           "moved_to_device_since": round(moved, 5),
                           "device": round(device, 5),
                           "sum": round(host + mixed + moved + device, 5)},
        "closure_first_cut": {
            "host_leaves_s": round(host, 5),
            "mixed_leaf_upper_bound_s": round(mixed, 5),
            "host_inside_device_units_s": HOST_INSIDE_DEVICE_UNITS,
            "named_total_s": [round(named_lo, 5), round(named_hi, 5)],
            "remainder_s": list(REMAINDER),
            "named_pct_of_remainder": [round(100 * named_lo / REMAINDER[1], 1),
                                       round(100 * named_hi / REMAINDER[0], 1)],
            "unnamed_s": [round(REMAINDER[0] - named_hi, 4), round(REMAINDER[1] - named_lo, 4)],
            "note": "a range because the `predict_step` leaf is an upper bound: part of it is "
                    "host z_init that the default-on device path deleted. The low end assumes "
                    "all of that leaf moved, the high end assumes none of it did.",
        },
        "reducible_first_cut": {
            "reducible_s": round(reducible, 5),
            "bar_s": REDUCIBLE_BAR,
            "fraction_of_bar": round(reducible / REDUCIBLE_BAR, 3),
            "shortfall_s": round(REDUCIBLE_BAR - reducible, 4),
            "extreme_fraction_of_bar": None,  # filled below, needs `mixed`
            "pipelined_service_only_s": tree["write_result"]["excl_s"],
            "pipelined_service_only_argument":
                "the CIF write, 0.0570 s. Real, and a THROUGHPUT win only: in a multi-target "
                "service target N's write overlaps target N+1's fold, which is how JapanFold "
                "serves. Against the 12.5 s latency cell it is worth nothing, because the timed "
                "region ends with the file on disk and deferring the write stops the timer "
                "earlier instead of finishing sooner.",
            "generous_ceiling_s": round(reducible + 0.5 * tree["prepare"]["excl_s"], 5),
            "generous_argument": "crediting half of prepare, which is not priced. Neither the "
                                 "CIF write nor half of predict_step is credited: the write is "
                                 "inside the timed region by definition, and half of "
                                 "predict_step is half of a number that is partly gone to the "
                                 "device already.",
            "extreme_ceiling_s": round(reducible + 0.5 * tree["prepare"]["excl_s"]
                                       + mixed, 5),
            "extreme_argument": "half of prepare AND the whole pre-port predict_step leaf "
                                "credited. This is NOT a bound on what the host can give: the "
                                "leaf's value today is unknown in BOTH directions. See "
                                "UNNAMED_IS_THE_ANSWER.",
            "UNNAMED_IS_THE_ANSWER": {
                "claim": "every host item this row can NAME is irreducible or device-only, and "
                         "the whole reducibility question collapses onto the 0.3997-0.7738 s "
                         "that no named item accounts for. That block is Boltz2.forward's own "
                         "body, and the pre-registered program-cache clear (0.0-2.3 s band) is "
                         "the candidate that could occupy most of it.",
                "why_the_mixed_leaf_is_not_an_upper_bound":
                    "a port does not only subtract. perf/b2z2_hostzero's two committed trees "
                    "differ by device-conditioning alone: turning it ON deleted diffusion_cond's "
                    "1.0665 s of children and RAISED `predict_step` exclusive from 0.46468 s to "
                    "0.72257 s, because the port adds the host glue that builds and uploads the "
                    "device module's inputs. Net host win, but the leaf GREW 1.55x. Those two "
                    "are Wormhole Galaxy at 40 s folds so the seconds do not transfer, and "
                    "`rel_pos` survives in both, which dates them to device_zinit OFF. Today on "
                    "qb2 both ports are on: zinit removes the host z_init build from that leaf "
                    "and conditioning adds upload glue to it. The sign of the net is unknown.",
                "hard_ceiling_that_does_not_depend_on_any_of_it":
                    "c12-profiled-fold measured the device term in situ, so ALL exposed host "
                    "time is 1.6489-1.6720 s however the leaf splits. Subtracting the 0.5610 s "
                    "of identified pure-host leaves and the 0.3372 s of host inside the device "
                    "units leaves at most 0.7738 s unnamed.",
                "worst_case_s": 0.8696,
                "worst_case_argument":
                    "half of prepare (0.0958 s) plus ALL 0.7738 s of unnamed time credited as "
                    "reducible is 0.8696 s, which would just clear the 0.8490 s bar. Nobody "
                    "should bank that -- it assumes every unnamed second is deletable glue -- "
                    "but it means this row must NOT claim the host cannot reach the bar. It "
                    "claims something narrower and better supported: nothing the row can name "
                    "reaches it, and one unmeasured item decides the rest.",
                "never_instrumented":
                    "the reason this block exists at all, and why no other row's data can "
                    "substitute: Boltz2.forward's OWN BODY has never been instrumented by "
                    "anyone. Every committed tree patches either predict_step, which is above "
                    "it, or the device modules, which are below it, so forward's body has always "
                    "fallen out as predict_step's exclusive remainder. Checked exhaustively: "
                    "b2x_host_residual's 128/512 on-card trees, its CPU-only hostpath_512_pc "
                    "(same six stages, no s_init / pair_mask / program-cache rows), the four "
                    "whglx 512 aa trees, and c12-profiled-fold's per-unit book. decomp.py's "
                    "install_extra is the first instrument that patches `forward`, which is what "
                    "its own docstring says it was added for.",
                "what_settles_it":
                    "one A/B on TT_BIO_BOLTZ2_KEEP_PROGRAM_CACHE inside a fold-repeating "
                    "process, plus brackets inside Boltz2.forward. That is the first thing the "
                    "capture should run, ahead of the two-clock arms.",
            },
            "fold_if_all_named_host_deleted_s": [round(FOLD_TODAY - named_hi, 4),
                                                 round(FOLD_TODAY - named_lo, 4)],
        },
        "rows": rows,
        "scaling_first_cut": scaling(tree),
        "instrument_perturbation": perturbation(),
    }
    r = out["reducible_first_cut"]
    r["extreme_fraction_of_bar"] = round(r["extreme_ceiling_s"] / REDUCIBLE_BAR, 3)
    r["extreme_shortfall_s"] = round(REDUCIBLE_BAR - r["extreme_ceiling_s"], 4)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
