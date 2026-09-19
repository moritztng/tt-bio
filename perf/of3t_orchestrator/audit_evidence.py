"""Check every headline number in the campaign's scoreboard against its committed artifact.

`state/of3t/EVIDENCE.md` is written by the orchestrator by TRANSCRIBING numbers out of row
reports, and a transcription can drift from the artifact it came from -- silently, because both
look like prose. This re-reads the artifacts on `wk/of3t` and asserts the figures the scoreboard
quotes. A mismatch is a finding about the scoreboard, not about the row.

It also pins the DENOMINATORS, which is this campaign's own standing rule (K29: report leaves
against total, never leaves alone). The tape's headline "11003 / 11003" is only honest if the
calls excluded from the denominator genuinely carry no input tensor, so that decomposition is
asserted here rather than taken on trust.

CPU only, no card, no network. Run from a `wk/of3t` checkout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ok, bad, warn = [], [], []


def j(rel):
    p = ROOT / rel
    if not p.is_file():
        bad.append(f"MISSING artifact {rel}")
        return None
    return json.loads(p.read_text())


def check(label, got, want, rel=None):
    if got == want:
        ok.append(f"{label}: {got}")
    else:
        bad.append(f"{label}: artifact says {got!r}, scoreboard says {want!r}"
                   + (f" ({rel})" if rel else ""))


def close(label, got, want, tol=5e-4):
    if got is not None and abs(got - want) <= tol * max(abs(want), 1e-30):
        ok.append(f"{label}: {got:.6g}")
    else:
        bad.append(f"{label}: artifact says {got!r}, scoreboard says {want!r}")


# --- tape coverage, and the denominator the "100 %" rests on -------------------------------
d = j("perf/of3t_tape/coverage_trunk_c1_b48_strict.json")
if d:
    t = d["totals"]
    check("tape trunk taped calls", t["taped"], 11003)
    # The claim is 11003 of 11003 CALLS THAT CARRY A TENSOR. That is only honest if every
    # excluded call is a tensor SOURCE or a non-math accessor. Assert the composition of the
    # excluded bucket, not just its size.
    untaped = {k.split("|")[0]: sum(v.values()) for k, v in d["sites"].items()
               if not k.endswith("|taped")}
    allowed = {"from_torch", "zeros", "get_memory_view", "linear"}   # linear -> nested
    stray = {k: n for k, n in untaped.items() if k not in allowed}
    if stray:
        bad.append(f"tape denominator: calls excluded that are NOT tensor sources: {stray}")
    else:
        ok.append(f"tape denominator honest: excluded = {untaped} "
                  f"(sources/accessors/nested only)")
    check("tape nested", t["nested"], 14)
    check("tape none", t["none"], 221)

# --- leaves against total, all five models -------------------------------------------------
for model, leaves, total in [("of3", 180, 204), ("protenix", 180, 196), ("boltz2", 180, 204),
                             ("boltzgen", 180, 204), ("af2_b4", 0, 184), ("af2_b1", 0, 46)]:
    f = f"perf/of3t_leaves/leaves_{model.replace('_b4','')}_b4_n64.json" if model != "af2_b1" \
        else "perf/of3t_leaves/leaves_af2_b1_n64.json"
    d = j(f)
    if d:
        check(f"leaves {model} with_grad", len(d["with_grad"]) if isinstance(d["with_grad"], list)
              else d["with_grad"], leaves, f)
        check(f"leaves {model} total", d["total"], total, f)

# --- bijection manifest --------------------------------------------------------------------
d = j("perf/of3t_equivalence/bijection_manifest.json")
if d:
    check("manifest their tensors", d["their_tensor_count"], 4935)
    check("manifest mapped in scope", d["coverage"]["their_tensors_mapped"], 3275)
    check("manifest unmapped in scope", d["coverage"]["their_tensors_unmapped_in_scope"], 0)
    check("manifest fused", d["fused_tensor_count"], 232)
    check("manifest out of scope", d["out_of_scope"]["count"], 1660)

# --- instruments B, C, clipping, trajectory -------------------------------------------------
d = j("perf/of3t_equivalence/instrument_b_lr.json")
if d:
    cfgs = d.get("configs", {})
    tot = sum(c.get("steps_compared", 0) for c in cfgs.values())
    mis = sum(c.get("mismatches", 0) for c in cfgs.values())
    check("schedule comparisons", tot, 109005)
    check("schedule mismatches", mis, 0)

d = j("perf/of3t_equivalence/instrument_c_optim.json")
if d:
    w = d["arms"]["constant_lr"]["worst"]
    close("optimizer worst rel", w["rel"], 2.738160842740276e-07)
    check("optimizer worst step", w["step"], 191)
    check("optimizer worst tensor", w["tensor"], "bias")

d = j("perf/of3t_orchestrator/instrument_b2_clip.json")
if d:
    close("clip disabled-param divergence (pre-fix)",
          d["disabled_params"]["rel_clip"], 8.368e-01, tol=2e-3)
    close("clip per-sample divergence (pre-fix)",
          d["per_sample_vs_batch"]["relative_difference"], 1.948e-01, tol=2e-3)

d = j("perf/of3t_equivalence/instrument_t_traj.json")
if d:
    check("trajectory N", d["N"], 20)
    check("trajectory verdict", d["verdict"], "PASS")

# --- the reference bundle: is it usable, and do its two manifests agree? ---------------------
# A bundle can be published, hashed and finite-difference validated and still be unusable: the
# BUNDLE-MIN gradient passed all three while 4,141 of 4,147 tensors were exactly zero, because
# the FD check sampled only entries where |analytic| >= 1e-6 and so could never look at them.
# Two manifests differing only in case made it worse -- the lowercase one still reports
# n_with_gradient 4147 and no hold. Assert the authoritative one and refuse to let a composition
# quietly carry a bundle whose own manifest says do not use it.
bdir = ROOT / "perf/of3t_reference/bundle_min"
upper, lower = bdir / "MANIFEST.json", bdir / "manifest.json"
if upper.is_file():
    m = json.loads(upper.read_text())
    st = str(m.get("status", ""))
    # PUBLISHED is a claim about the artifacts, not about the manifest. A status line saying
    # PUBLISHED while the gradient it names is still "in transfer" sends a consumer to a path
    # that has a .part file on it -- the R43 shape again, one pass after R43 was fixed. Check
    # the declared per-file presence, not just the headline.
    arts = m.get("artifacts", [])
    if isinstance(arts, list) and "PUBLISH" in st.upper():
        absent = [a.get("file") for a in arts
                  if isinstance(a, dict) and a.get("on_qb2") not in (True, None)]
        if absent:
            warn.append(f"bundle manifest says PUBLISHED but declares these artifacts not on "
                        f"the host: {', '.join(str(x) for x in absent[:4])}"
                        + (" ..." if len(absent) > 4 else ""))
        else:
            ok.append("bundle says PUBLISHED and declares every artifact present")
    if "HOLD" in st.upper():
        # A held DATA artifact does not make the COMPOSITION wrong -- no claim is being made
        # against it -- so this warns loudly rather than failing, the same way an unpushed
        # worktree does. It becomes a failure the moment a row quotes a number from it.
        warn.append(f"reference bundle is ON HOLD, do not compare against it: {st[:110]}")
    else:
        ok.append("reference bundle manifest carries no hold")
    if lower.is_file():
        lo = json.loads(lower.read_text())
        if "status" not in lo:
            warn.append("two manifests differ only in case and the lowercase one carries no "
                        "status field -- a consumer reading manifest.json gets a stale 'valid' "
                        "bundle. Delete it or make it a pointer")

# --- CROSS-INSTRUMENT CONSISTENCY -----------------------------------------------------------
# Each instrument can be internally correct and still disagree with its neighbour about a fact
# both depend on. Nothing in this campaign caught instrument B configuring OF3's schedule at
# max_lr 1e-3 while instrument C drove the optimizer at 1.8e-3, because each passed on its own.
# --- of3t-entity: D16, the two conventions, and a control that predicted its own number ----
d = j("perf/of3t_entity/mapping.json")
if d:
    c = d["conventions"]
    check("D16 af3 convention", (c["af3"]["protein"], c["af3"]["rna"], c["af3"]["dna"],
                                 c["af3"]["ligand"]), (0, 1, 2, 3))
    check("D16 boltz convention", (c["boltz"]["protein"], c["boltz"]["rna"], c["boltz"]["dna"],
                                   c["boltz"]["ligand"]), (0, 2, 1, 3))
    check("D16 dna/rna are swapped between them",
          d["dna_and_rna_are_swapped_between_the_two_conventions"], True)
    # The table is only evidence if every stack was EXECUTED and matched what it declares.
    check("D16 every stack matches its own declared source",
          d["every_stack_matches_its_own_declared_source"], True)

d = j("perf/of3t_entity/entity_delta.json")
if d:
    s = d["delta_moltype_stacks"]
    close("D16 boltz-2 value delta", s["boltz-2"]["delta"]["rel_value"], 0.869740276433419)
    close("D16 boltz-2 seed delta", s["boltz-2"]["delta"]["rel_seed"], 0.9008886419943194)
    close("D16 boltzgen value delta", s["boltzgen"]["delta"]["rel_value"], 0.8651116903880804)
    close("D16 boltzgen seed delta", s["boltzgen"]["delta"]["rel_seed"], 0.8981038214816643)
    # The two conventions disagree in the DATA and not only in the tables, and the loss is
    # identical anyway. Both halves are load-bearing: the first says the swap is real, the
    # second says an unnamed convention is safe TODAY, for a reason that can expire.
    dd = s["_convention_disagreement_in_the_data"]
    check("D16 columns differ elementwise", dd["columns_elementwise_equal"], False)
    check("D16 tokens that differ", dd["n_tokens_that_differ"], 16)
    check("D16 they differ only on nucleic acid", dd["they_differ_only_on_nucleic_acid_tokens"], True)
    check("D16 loss identical anyway (w_dna == w_rna)", dd["loss_is_identical_anyway"], True)
    check("D16 AF2 weighting is vacuous, measured", d["af2"]["weighting_is_vacuous"], True)
    check("D16 AF2 all-protein column is bit-identical to bare",
          d["af2"]["all_protein_mol_type"]["bit_identical_to_bare"], True)
    # The control's whole value is that the prediction existed BEFORE the measurement.
    ctl = d["control"]
    check("D16 control ligand tokens non-zero", ctl["n_ligand_tokens"], 31)
    close("D16 control measured value ratio", ctl["perturbation"]["measured_value_ratio"],
          0.6288281900980327, tol=1e-12)
    ad = ctl["absent_flag_detector"]
    check("D16 zeroed flags bit-identical to absent", ad["zeroed_is_bit_identical_to_absent"], True)

d = j("perf/of3t_entity/census.json")
if d:
    for arm in ("mol_type_af3", "mol_type_boltz", "mol_type_unnamed"):
        check(f"D16 derived matches native ({arm})", d["derived_matches_native"][arm], True)
    close("D16 mse value under the af3 arm", d["mse_across_arms"]["mol_type_af3"]["value"],
          1.407684903716911)

# --- D17 on a card, and the ceiling it puts on every SS3d number ---------------------------
d = j("perf/of3t_gradients/reach_by_norm.json")
if d:
    r = d["reach"]
    check("D17 reference tensors", d["n_tensors"], 4147)
    check("D17 none absent", d["n_absent"], 0)
    close("D17 tracer reach by norm", r["k22_tracer_bijection"]["norm_share"], 0.06543442172265604)
    close("D17 device reach by norm", r["device_bijection_mat64"]["norm_share"], 0.9800944410036996)
    check("D17 device tensors mapped", r["device_bijection_mat64"]["tensors"], 3545)
    # The published global norm is the one number that ties this artifact to the manifest.
    close("D17 global norm vs manifest", d["total_norm"], 3.908301894520238, tol=1e-12)
    # The ceiling. These two are what every SS3d figure in EVIDENCE is measured inside.
    close("D17 block-0 share of the squared norm",
          r["instrument_a_block0_53"]["norm_share"], 0.0020038559007500654)
    close("D17 whole-trunk share of the squared norm",
          r["pairformer_stack_all"]["norm_share"], 0.05274678966027111)
    if r["pairformer_stack_all"]["norm_share"] < 0.10:
        ok.append(f"D17 ceiling stands: the whole 48-block trunk is "
                  f"{r['pairformer_stack_all']['norm_share']:.2%} of the squared norm, block 0 "
                  f"is {r['instrument_a_block0_53']['norm_share']:.2%}")
    else:
        bad.append("D17's ceiling has moved -- EVIDENCE's scope note says no trunk-scope "
                   "instrument can speak for the gradient's magnitude, and that sentence is "
                   "built on the trunk holding ~5 % of it")

# --- D14's crop ladder: both units at every rung, and the 384 refusal's own arithmetic -----
_rungs = {}
for _n in (128, 256, 384):
    _d = j(f"perf/of3t_l1/out/r2_{_n}.json")
    if _d:
        _rungs[_n] = _d
if len(_rungs) == 3:
    for _n, _want_fwd, _want_bwd, _ok in ((128, 1696363520, 3929519104, True),
                                          (256, 2929000000, 14755173376, True),
                                          (384, 4795000000, 32369505280, False)):
        _b = _rungs[_n]["backward"]
        check(f"D14 rung {_n} backward fits", _b["ok"], _ok)
        close(f"D14 rung {_n} backward DRAM peak", _b["dram_peak_b"], _want_bwd, tol=1e-3)
    # The claim that 384 is FRAGMENTATION and not capacity rests on one comparison: the
    # requested block against the largest free one. If the request ever stops being the
    # largest allocation, the diagnosis changes and so does the remedy.
    _req = _rungs[384]["backward"].get("dram_largest_b")
    if _req == 1207959552:
        ok.append("D14 the 384 request is 1207959552 B -- the fragmentation diagnosis is about "
                  "THIS allocation, 3.26 MB above the largest contiguous block")
    else:
        bad.append(f"D14's 384 request is now {_req} B, not 1207959552 -- the '3.26 MB short' "
                   f"figure is about a block size that has changed")
    # Both units, every rung (R16's lesson). A rung reported in one unit hides which limit bound it.
    if all(_rungs[_n]["backward"].get("dram_live_allocs") for _n in (128, 256, 384)):
        ok.append("D14 ladder reports allocations AND bytes at every rung: "
                  + ", ".join(f"{_n}:{_rungs[_n]['backward']['dram_live_allocs']}"
                              for _n in (128, 256, 384)))
    else:
        bad.append("D14 ladder is missing a live-allocation count at some rung -- one unit "
                   "hides which limit bound it")

d = j("perf/of3t_l1/out/gradient_control.json")
if d:
    # This artifact is the CONTROL (a perturbed run), so FAIL is its correct verdict. The
    # campaign's claim is the unperturbed comparison; what is asserted here is that the
    # control discriminates at all.
    check("D14 control compared", d["compared"], 172)
    check("D14 control set identical", (d["n_only_in_a"], d["n_only_in_b"]), (0, 0))
    if d["worst_rel_l2"] > d["bar_per_tensor"]:
        ok.append(f"D14 one-bf16-ulp control is rejected at {d['worst_rel_l2']:.3f} on "
                  f"{d['worst_tensor'].split('.')[-1]}, {d['worst_rel_l2']/d['bar_per_tensor']:.1f}x the bar")
    else:
        bad.append("D14's one-ulp control no longer fails -- a gate nobody has watched fail is "
                   "not a gate, and the bit-identity result leans on this one discriminating")

# --- of3t-gradients: instrument A at block scope, and the reference that disqualified itself
d = j("perf/of3t_gradients/instrument_a_bundle_block0.json")
if d:
    s = d["summary"]
    close("A block0 median", s["median"], 0.0781291094815586)
    check("A block0 median passes its bar", s["median_pass"], False)
    check("A block0 per-tensor passes its bar", s["per_tensor_pass"], False)
    check("A block0 compared", s["compared"], 53)
    check("A block0 absent (fused qkv)", s["absent"], 4)
    close("A block0 forward s, real tokens", d["forward_rel"]["s_masked"], 0.008061467639837464)
    close("A block0 forward z, real tokens", d["forward_rel"]["z_masked"], 0.007978705045610828)
    # A14(i): the worst figure is quoted WITHOUT the degenerate denominator, and the scoreboard
    # says 0.952. Recompute the split here rather than trusting either number in isolation --
    # if a second tensor ever falls under the floor, the headline changes and this says so.
    pp = d.get("per_parameter", [])
    FLOOR = 1e-12
    tiny = [q for q in pp if q.get("ref_norm") is not None and q["ref_norm"] < FLOOR]
    rest = [q["rel_l2"] for q in pp if q.get("ref_norm") is not None and q["ref_norm"] >= FLOOR]
    if len(tiny) == 1 and abs(max(rest) - 0.9519554376602173) <= 1e-2 * 0.952:
        ok.append(f"A14 denominator floor: 1 tensor under 1e-12 (ref_norm "
                  f"{tiny[0]['ref_norm']:.3g}), worst of the rest {max(rest):.4g}")
    else:
        bad.append(f"A14 split moved: {len(tiny)} tensor(s) under the {FLOOR:g} floor, worst of "
                   f"the rest {max(rest) if rest else float('nan'):.4g}. The scoreboard quotes "
                   f"1 and 0.952 -- a relative computed on a reference norm below its own "
                   f"population's scale is not a measurement")
    # A14(ii): the protocol's own 1 % control did NOT fire here. If that ever flips, the
    # amendment's justification is gone and the row should be re-read, not silently trusted.
    nc = d.get("negative_control", {})
    if nc.get("protocol_1pct", {}).get("fires") is False and nc.get("calibrated", {}).get("fires"):
        ok.append("A14 control sizing: the protocol's 1 % does not fire at this baseline, the "
                  "calibrated x1.10 does")
    else:
        warn.append("A14's premise has changed -- the 1 % control now fires, so the sizing "
                    "amendment needs re-reading against this artifact")

d = j("perf/of3t_gradients/dropout_floor_block0.json")
if d:
    on1 = d["dropout_on_pairs"]["1_vs_2"]
    off = d["dropout_off_control"]
    close("D18 dropout-on worst (seeds 1 vs 2)", on1["worst"], 1.166879500823583)
    close("D18 dropout-on median (seeds 1 vs 2)", on1["median"], 0.5499371745996952)
    check("D18 dropout-on over bar", on1["over_bar"], 47)
    # The whole identification rests on this being EXACTLY zero, twice. Not "small".
    check("D18 dropout-off worst is exactly 0", off["worst"], 0.0)
    check("D18 dropout-off median is exactly 0", off["median"], 0.0)
    check("D18 dropout-off over bar", off["over_bar"], 0)
    if on1["median"] > 10 * d["bar"]:
        ok.append(f"D18 the reference's own floor ({on1['median']:.3f}) is "
                  f"{on1['median']/d['bar']:.0f}x the bar it would be judged at ({d['bar']})")
    else:
        warn.append("D18's floor no longer exceeds its bar by an order of magnitude -- if the "
                    "bundle was republished, this check and EVIDENCE's row both need rewriting")

# --- of3t-updaterule: D11, D12, and the bijection's true reach (D17) -----------------------
d = j("perf/of3t_updaterule/lr_wiring.json")
if d:
    check("D11 closed-form points exact", d["arm_a_closed_form"]["exact_matches"], 1007)
    check("D11 closed-form mismatches", d["arm_a_closed_form"]["mismatches"], 0)
    check("D11 applied-rate steps exact", d["arm_b_wiring"]["exact_matches"], 2005)
    check("D11 applied-rate mismatches", d["arm_b_wiring"]["mismatches"], 0)
    for arm in ("shipped_warmup_1000", "scaled_warmup_20"):
        a_ = d["arm_c_trajectory"][arm]
        check(f"D11 d_1 zero both sides ({arm})", a_["d1_zero_both_sides"], True)
        # The rung the whole defect turns on. `d1_ours` is a float and 0.0 is the claim.
        check(f"D11 d_1 ours ({arm})", a_["d1_ours"], 0.0)
    close("D11 worst d_k, shipped warmup",
          d["arm_c_trajectory"]["shipped_warmup_1000"]["worst_rel_d_k2_20"], 2.0523076682188713e-06)
    # A control that did not break the arms it should have proves nothing (SS3e).
    check("D11 control puts d_1 back", d["negative_control"]["c_d1_ours"] > 0, True)
    close("D11 control worst d_k", d["negative_control"]["c_worst_rel_d_k2_20"], 2.170127574298317)

d = j("perf/of3t_updaterule/mse_entity.json")
if d:
    close("D12 value delta", d["arm_delta"]["rel_value"], 0.23344546565807414)
    close("D12 gradient-seed delta", d["arm_delta"]["rel_seed"], 0.7636994469305252)
    close("D12 vs their mse_loss, value", d["arm_reference"]["rel_value"], 6.057483991567378e-16,
          tol=1e-2)
    close("D12 vs their mse_loss, gradient", d["arm_reference"]["rel_grad"],
          3.662141705890694e-15, tol=1e-2)
    # The row's own sharpest finding: zeroed flags and absent flags are numerically identical,
    # so `without` is the only detector. If that ever stops holding, the claim changes shape.
    z = d["arm_control"]["flags_zeroed"]
    check("D12 zeroed flags == dropped flags (value)", z["value_equals_dropped"], True)
    check("D12 zeroed flags == dropped flags (seed)", z["seed_equals_dropped"], True)

d = j("perf/of3t_updaterule/reference_profile.json")
if d:
    b_ = d["bijection_split"]
    check("D17 reference tensors with a gradient", b_["their_tensors_with_a_gradient"], 4147)
    check("D17 tensors mapped by the bijection", b_["mapped_by_the_bijection"], 3275)
    close("D17 fraction of gradient NORM reachable", b_["fraction_of_gradient_norm_reachable"],
          0.06543442172265605)
    check("D17 reference presence set agrees", d["presence"]["set_agrees"], True)
    check("D17 reference sha256 verified", d["sha256_matches"], True)
    # Count and norm must both be quoted, always. 79 % and 6.5 % are the same split.
    frac_count = b_["mapped_by_the_bijection"] / b_["their_tensors_with_a_gradient"]
    if frac_count - b_["fraction_of_gradient_norm_reachable"] > 0.5:
        ok.append(f"D17 count/norm gap intact: {frac_count:.0%} of tensors, "
                  f"{b_['fraction_of_gradient_norm_reachable']:.2%} of the squared norm")
    else:
        warn.append("D17's count/norm gap has closed -- re-read the scoreboard row, it is "
                    "written to explain a gap that no longer exists")

# --- PROTOCOL A12: the SEAM between SS4 and SS5 ---------------------------------------------
# SS4 proves the schedule as a FUNCTION (upstream's real scheduler against `af3_lr`, exact, over
# 109,005 steps). SS5 proves the optimizer trajectory. Neither proves the MAPPING from update
# number to schedule argument, and D11 lived in that gap for thirty passes while both instruments
# read PASS -- because SS5 gated on OUR mapping and printed upstream's as "info".
#
# So the mapping is established here by RUNNING upstream's objects, not by reading source: drive
# a real `torch.optim.Adam` through a real `AlphaFoldLRScheduler` and record which lr each update
# actually multiplies by. Then require instrument C to be gated on that arm. Unavailable imports
# are REPORTED, never skipped silently -- a seam check that quietly does not run is the defect it
# exists to catch.
c = j("perf/of3t_equivalence/instrument_c_optim.json")
if c:
    arms = c.get("arms", {})
    if "of3_schedule_upstream" in arms and arms["of3_schedule_upstream"].get("pass"):
        ok.append("SS5 gates on upstream's index mapping (A12), and it passes")
    elif "of3_schedule_aligned" in arms:
        bad.append("SS5 gates on `of3_schedule_aligned`, which is OUR pre-D11 mapping (update k "
                   "at lr(k)). That arm agreeing proves the optimizer, not the alignment -- see "
                   "DEFECTS D15, PROTOCOL A12")
    else:
        bad.append(f"SS5 has no gated upstream-alignment arm; arms present: {sorted(arms)}")
    if "of3_schedule_offset_by_one" not in arms:
        warn.append("SS5 no longer reports the offset arm -- the size of a one-step schedule "
                    "error is how a reader calibrates the gated number")

try:
    import torch
    from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
    _p = torch.nn.Parameter(torch.zeros(1))
    _o = torch.optim.Adam([_p], lr=1.8e-3)
    _s = AlphaFoldLRScheduler(_o, max_lr=1.8e-3, warmup_no_steps=1000)
    seen = []
    for _ in range(5):
        seen.append(_o.param_groups[0]["lr"])
        _p.grad = torch.ones(1); _o.step(); _s.step()
    want = [0.0, 1.8e-06, 3.6e-06, 5.4e-06, 7.2e-06]
    if all(abs(g - w) <= 1e-12 for g, w in zip(seen, want)):
        ok.append("upstream mapping measured live: update k runs at lr(k-1), update 1 at exactly 0")
    else:
        bad.append(f"upstream's own scheduler now yields {seen} at updates 1..5, not {want}. "
                   f"A12's index mapping is derived from THIS measurement, so every SS5 verdict "
                   f"is suspect until it is re-derived")
except Exception as e:                                   # noqa: BLE001
    warn.append(f"A12 live mapping probe did not run ({type(e).__name__}: {e}) -- the SS4/SS5 "
                f"seam is unchecked on this run, which is exactly how D11 survived")

# Upstream settles it: `runner.py:863` builds the scheduler with
# `max_lr=optimizer_config.learning_rate`, so the two are THE SAME NUMBER by construction and
# any instrument pair that disagrees has one of them wrong.
b = j("perf/of3t_equivalence/instrument_b_lr.json")
c = j("perf/of3t_equivalence/instrument_c_optim.json")
if b and c:
    b_lr = b.get("configs", {}).get("of3_defaults", {}).get("config", {}).get("max_lr")
    c_lr = c.get("their_config", {}).get("learning_rate")
    if b_lr == c_lr:
        ok.append(f"cross-instrument lr agrees: B max_lr == C lr == {b_lr}")
    else:
        bad.append(f"cross-instrument lr DISAGREES: instrument B's 'of3_defaults' max_lr is "
                   f"{b_lr} while instrument C drives the optimizer at {c_lr}. Upstream ties "
                   f"them (runner.py:863, max_lr=optimizer_config.learning_rate), so one is "
                   f"wrong -- 1e-3 is the scheduler's Python signature default, not what OF3 "
                   f"ships")

# --- ONE reference, campaign-wide ----------------------------------------------------------
# `of3t-reference` is reopened to republish BUNDLE-MIN with Dropout in eval (D18), which will
# change `grads_f64_recycles0.pt`'s sha256. Four rows cite the old digest inside their own
# artifacts, and a rebuild that lands while any of them still quotes the old one leaves the
# campaign holding TWO references at once -- silently, because each file is internally
# consistent. So: every artifact that names the validated gradient's digest must name the SAME
# one, and it must be the one the MANIFEST declares.
_man = j("perf/of3t_reference/bundle_min/MANIFEST.json")
if _man:
    _declared = None
    for _a in _man.get("artifacts", []):
        if isinstance(_a, dict) and _a.get("file") == "grads_f64_recycles0.pt":
            _declared = _a.get("sha256")
    _citers = {
        "perf/of3t_updaterule/reference_profile.json": ("declared_sha256",),
        "perf/of3t_gradients/reach_by_norm.json": ("reference", "sha256"),
        "perf/of3t_gradients/instrument_a_bundle_block0.json": ("bundle", "sha256"),
    }
    _seen = {}
    for _rel, _path in _citers.items():
        _d = j(_rel)
        if not _d:
            continue
        _v = _d
        for _k in _path:
            _v = _v.get(_k) if isinstance(_v, dict) else None
        if _v:
            _seen[_rel] = _v
    if _declared and _seen:
        _bad = {k: v for k, v in _seen.items() if not _declared.startswith(str(v)[:40])
                and not str(v).startswith(_declared[:40])}
        if _bad:
            bad.append(f"TWO REFERENCES IN FLIGHT: the MANIFEST declares "
                       f"{_declared[:16]}... for grads_f64_recycles0.pt while "
                       + "; ".join(f"{k} cites {v[:16]}..." for k, v in _bad.items())
                       + ". Every figure measured against the other one is a figure about an "
                         "artifact that no longer exists (D18)")
        else:
            ok.append(f"one reference campaign-wide: {len(_seen)} artifacts and the MANIFEST "
                      f"all cite {_declared[:16]}...")
    elif not _declared:
        warn.append("the MANIFEST no longer declares a sha256 for grads_f64_recycles0.pt -- "
                    "the one-reference check cannot run, which is how D18 stayed invisible")

# --- every UNFIXED defect must be named in the orchestrator's GAP ----------------------------
# GAP has drifted twice: it described the campaign as it stood seven passes earlier, and then
# omitted the hardest blocker entirely. The gate only checks that the field EXISTS. A summary
# written by transcription drifts exactly like a scoreboard does, so it gets the same treatment
# as the scoreboard: checked against its source.
DEF = Path("/home/moritz/.coworker/state/of3t/DEFECTS.md")
ORCH = Path("/home/moritz/.coworker/state/of3t-orchestrator.md")
if DEF.is_file() and ORCH.is_file():
    import re as _re
    unfixed = [m.group(1) for m in
               _re.finditer(r"^### (D\d+)\..*$", DEF.read_text(), _re.M)
               if "UNFIXED" in m.group(0)]
    o = ORCH.read_text()
    g = _re.search(r"^GAP:(.*?)(?=^VERDICT:)", o, _re.M | _re.S)
    gap = g.group(1) if g else ""
    missing = [d for d in unfixed if not _re.search(rf"\b{d}\b", gap)]
    if missing:
        bad.append(f"GAP does not name these UNFIXED defects: {', '.join(missing)} "
                   f"-- the summary has drifted from DEFECTS.md")
    else:
        ok.append(f"GAP names all {len(unfixed)} UNFIXED defects")

# --- PROVES / DOESNOT are transcription too, and they are what Moritz reads ------------------
# EVIDENCE.md is audited figure by figure; the two summary fields quote the same numbers in
# prose and were, until pass 44, unchecked -- one of them still said "amended nine times" at
# fifteen amendments and "instrument A has never run" after it had run and failed. A summary
# written by transcription drifts exactly like a scoreboard, so it gets the same treatment.
# Each entry is (artifact-derived string, where it must appear, why it is load-bearing).
if ORCH.is_file():
    import re as _re
    o = ORCH.read_text()
    pr = _re.search(r"^PROVES:(.*?)(?=^DOESNOT:)", o, _re.M | _re.S)
    dn = _re.search(r"^DOESNOT:(.*?)(?=^GAP:)", o, _re.M | _re.S)
    proves, doesnot = (pr.group(1) if pr else ""), (dn.group(1) if dn else "")
    both = proves + doesnot
    claims = []
    _r = j("perf/of3t_gradients/reach_by_norm.json")
    if _r:
        share = _r["reach"]["device_bijection_mat64"]["norm_share"]
        claims.append((f"{share*100:.2f} %", proves, "the bijection's reach"))
        claims.append((f"{_r['reach']['pairformer_stack_all']['norm_share']*100:.2f} %",
                       doesnot, "the trunk's share of the norm, which bounds every SS3d claim"))
    _c = j("perf/of3t_equivalence/instrument_c_optim.json")
    if _c and "of3_schedule_upstream" in _c.get("arms", {}):
        claims.append((f"{_c['arms']['of3_schedule_upstream']['worst']['rel']:.3e}",
                       proves, "SS5 against upstream's alignment"))
    _a = j("perf/of3t_gradients/instrument_a_bundle_block0.json")
    if _a:
        claims.append((f"{_a['summary']['median']:.3e}", doesnot,
                       "instrument A's block-0 median, the one FAILING number"))
    _d = j("perf/of3t_orchestrator/instrument_c2_clip_in_step.json")
    if _d:
        claims.append((f"{_d['arms']['clip_binds']['worst']['rel']:.3e}", proves,
                       "the clip/optimizer seam"))
    missing = [(s, why) for s, where, why in claims if s not in where]
    if missing:
        for s, why in missing:
            bad.append(f"PROVES/DOESNOT does not quote {s} ({why}) -- the artifact has moved "
                       f"and the summary Moritz reads has not")
    elif claims:
        ok.append(f"PROVES/DOESNOT quote all {len(claims)} audited figures as the artifacts "
                  f"have them")
    # And the amendment count, which is a claim about the protocol's own history.
    _pp = Path("/home/moritz/.coworker/state/of3t/PROTOCOL.md")
    if _pp.is_file():
        n_am = len(_re.findall(r"^\*\*A\d+ \u2014", _pp.read_text(), _re.M))
        words = {9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
                 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
                 18: "eighteen", 19: "nineteen", 20: "twenty"}
        w = words.get(n_am)
        if w and _re.search(r"\b(nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                            r"seventeen|eighteen|nineteen|twenty) amendments\b", both):
            if f"{w} amendments" in both:
                ok.append(f"PROVES states the amendment count correctly ({w}, {n_am})")
            else:
                bad.append(f"PROVES states the wrong amendment count -- PROTOCOL has {n_am} "
                           f"({w})")

print("AUDIT of state/of3t/EVIDENCE.md against committed artifacts\n")
for line in ok:
    print(f"  ok    {line}")
for line in warn:
    print(f"  WARN  {line}")
for line in bad:
    print(f"  DRIFT {line}")
print(f"\n{len(ok)} confirmed, {len(warn)} warning(s), {len(bad)} drifted")
sys.exit(1 if bad else 0)
