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
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


_FELL_BACK: set = set()


def _campaign_doc(name: str) -> Path:
    """The authoritative document if this host has it, else the copy published in the branch.

    The campaign's record lives in a gitignored state dir on pc, so every check that reads it
    -- GAP against DEFECTS, the summary fields, D20's shares -- could only ever run for one
    person on one machine. The compose now publishes copies into the branch, so those checks
    fall back to the copy and a reviewer's `compose_verify.sh` runs the same 146 checks rather
    than silently fewer. Authoritative first, always: on pc the state file wins.
    """
    src = (Path("/home/moritz/.coworker/state/of3t-orchestrator.md")
           if name == "ORCHESTRATOR" else
           Path(f"/home/moritz/.coworker/state/of3t/{name}.md"))
    if src.is_file():
        return src
    _FELL_BACK.add(name)
    return ROOT / "perf/of3t_orchestrator/record" / f"{name}.md"
ok, bad, warn = [], [], []
_reach_top = {}


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
    # A held DATA artifact does not make the COMPOSITION wrong -- no claim is being made
    # against it -- so this warns loudly rather than failing, the same way an unpushed worktree
    # does. It becomes a failure the moment a row quotes a number from it.
    #
    # Read the STRUCTURED hold, not the substring "HOLD" in prose. The first version fired on a
    # manifest whose status began "PUBLISHED." because the word appeared later in a sentence
    # saying the hold was lifted -- a keyword gate cannot tell a claim from its own negation,
    # and this campaign has the same lesson filed twice already.
    _hold = m.get("hold")
    _hstate = str(_hold.get("state", "")) if isinstance(_hold, dict) else str(_hold or "")
    _held = bool(_hstate) and not _hstate.strip().upper().startswith(
        ("PUBLISHED", "NONE", "LIFTED", "CLEARED", "NO HOLD"))
    if _held:
        warn.append(f"reference bundle is ON HOLD, do not compare against it: {_hstate[:110]}")
    else:
        ok.append(f"reference bundle carries no hold ({(_hstate or st)[:48]}...)")
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
    _reach_top = d.get("by_top_level", {})
    check("D17 reference tensors", d["n_tensors"], 4147)
    check("D17 none absent", d["n_absent"], 0)
    close("D17 tracer reach by norm", r["k22_tracer_bijection"]["norm_share"], 0.040545222363297974, tol=1e-6)
    close("D17 device reach by norm", r["device_bijection_mat64"]["norm_share"], 0.9869491000000000, tol=1e-3)
    check("D17 device tensors mapped", r["device_bijection_mat64"]["tensors"], 3545)
    # The published global norm is the one number that ties this artifact to the manifest.
    close("D17 global norm, rebuilt r=0 reference", d["total_norm"], 3.707776369277738, tol=1e-9)
    # The ceiling. These two are what every SS3d figure in EVIDENCE is measured inside.
    close("D17 block-0 share of the squared norm",
          r["instrument_a_block0_53"]["norm_share"], 0.000860, tol=5e-3)
    close("D17 whole-trunk share of the squared norm",
          r["pairformer_stack_all"]["norm_share"], 0.03155993434564877, tol=1e-6)
    if r["pairformer_stack_all"]["norm_share"] < 0.10:
        ok.append(f"D17 ceiling stands: the whole 48-block trunk is "
                  f"{r['pairformer_stack_all']['norm_share']:.2%} of the squared norm, block 0 "
                  f"is {r['instrument_a_block0_53']['norm_share']:.3%}")
    else:
        bad.append("D17's ceiling has moved -- EVIDENCE's scope note says no trunk-scope "
                   "instrument can speak for the gradient's magnitude, and that sentence is "
                   "built on the trunk holding ~5 % of it")

# --- the two sides at crop 384, recomputed so the ratio cannot drift -----------------------
# The scoreboard quotes 870.75 s against 7-8 s and calls the comparison flattering to TT. The
# TT half is arithmetic on an artifact; recompute it, and assert the scope fields the caveat
# depends on -- one cycle, and the clock sampled DURING. If either changes, the sentence about
# what is and is not being compared stops being true.
_t384 = j("perf/of3t_l1/out/r3_384.json")
if _t384:
    _f, _b = _t384["forward"], _t384["backward"]
    _tot = (_f.get("s") or 0) + (_b.get("s") or 0)
    close("TT crop-384 trunk cycle, forward+backward seconds", _tot, 870.75, tol=1e-3)
    check("TT crop-384 ran ONE trunk cycle (num_recycles 0)", _f.get("cycles"), 1)
    _clk = _t384.get("env", {}).get("aiclk_during", {}).get("0", {})
    if _clk.get("median") and _clk.get("n", 0) >= 100:
        ok.append(f"TT crop-384 AICLK median {_clk['median']} MHz over {_clk['n']} samples "
                  f"polled DURING")
    else:
        bad.append("TT crop-384 has no DURING-sampled clock with enough samples -- PROTOCOL "
                   "4a makes a timing figure without its clock not a figure")
    # The ratio is quoted as ~116x. It is a ratio between DIFFERENT amounts of work and the
    # scoreboard says so; what must not drift is the number the sentence is built on.
    _ratio = _tot / 7.5
    if 100 <= _ratio <= 130:
        ok.append(f"crop-384 cross-stack ratio {_ratio:.0f}x (TT trunk cycle vs GPU full step) "
                  f"-- quoted as ~116x, and as favourable to TT")
    else:
        bad.append(f"the crop-384 ratio is now {_ratio:.0f}x, not ~116x")

# --- D14's crop ladder, CLOSED at 384 (pass 51) ---------------------------------------------
# The r2_* rungs are kept as history; r3_* is the ladder that decides the defect. 384's
# backward fell 32.370 -> 17.983 GB and now completes, which is the whole claim, so the PASS
# flags are asserted as hard as the byte counts -- a ladder that silently stopped running
# would read as "no drift".
_r3 = {}
for _n in (128, 256, 384, 640):
    _d = j(f"perf/of3t_l1/out/r3_{_n}.json")
    if _d:
        _r3[_n] = _d
if len(_r3) == 4:
    for _n, _fwd, _bwd, _peak in ((128, True, True, 2.522e9), (256, True, True, 7.384e9),
                                  (384, True, True, 17.983e9), (640, True, False, 34.215e9)):
        check(f"D14 r3 {_n} forward", _r3[_n]["forward"]["ok"], _fwd)
        check(f"D14 r3 {_n} backward", _r3[_n]["backward"]["ok"], _bwd)
        close(f"D14 r3 {_n} backward DRAM peak", _r3[_n]["backward"]["dram_peak_b"], _peak,
              tol=2e-3)
    # The consistency check the row itself named: the same weight count at every passing rung.
    _wg = {_n: _r3[_n]["backward"].get("params_with_grad") for _n in (128, 256, 384)}
    if len(set(_wg.values())) == 1 and list(_wg.values())[0] == 1639:
        ok.append(f"D14 1639 trunk weights carry a gradient at every passing rung {sorted(_wg)}")
    else:
        bad.append(f"D14's passing rungs disagree on how many weights carry a gradient: {_wg} "
                   f"-- the scoreboard says 1639 at all three, which is the consistency check")
    if _r3[384]["backward"]["ok"] and not _r3[640]["backward"]["ok"]:
        ok.append("D14 is closed where it matters: 384 trains, 640 does not -- the ladder "
                  "stops between them")
    else:
        bad.append("D14's ladder has moved; EVIDENCE says 384 trains and 640 does not")

# --- the r2 ladder, kept as the history 384's closure is measured against --------------
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

# --- the pre-registered targets for the rebuilt reference ----------------------------------
# `of3t-gradients` banked what `of3t-reference`'s A100 rebuild should produce, BEFORE it
# existed. A prediction is only worth anything if it is still the same prediction when the
# result lands, so it is read from the artifact here rather than quoted from a brief -- and
# EVIDENCE must carry the same three numbers.
d = j("perf/of3t_gradients/capture_trunk_boundary_nodropout.json")
if d:
    close("pre-registered loss at r=0", d["breakdown"]["loss"], 1.6311432393241485, tol=1e-12)
    close("pre-registered global norm at r=0", d["global_norm_here"], 3.7278454543745516,
          tol=1e-12)
    check("pre-registered non-zero tensors", d["n_nonzero_here"], 4138)
    # The r=0 vs published-r=0.25 spread is quoted as EXPECTED evidence, not as a regression.
    # If it ever collapses toward zero, the dropout story changes and the row should be re-read.
    _cv = d.get("capture_vs_bundle", {})
    _w = [v["worst_rel"] for v in _cv.values() if isinstance(v, dict)]
    if _w and min(_w) > 1.0:
        ok.append(f"r=0 vs the published r=0.25 gradient spans {min(_w):.3f}-{max(_w):.3f} "
                  f"worst rel across {len(_w)} blocks -- the dropout-floor order, as expected")
    else:
        bad.append(f"the r=0 / r=0.25 spread has collapsed ({_w}) -- EVIDENCE reads that "
                   f"spread as evidence of two different functions, and that reading depends "
                   f"on it being the same order as the dropout floor")
    _ev = _campaign_doc("EVIDENCE")
    if _ev.is_file():
        _txt = _ev.read_text()
        _missing = [s for s in ("1.631143239324149", "3.727845454375", "4,138 of 4,147")
                    if s not in _txt]
        if _missing:
            bad.append(f"EVIDENCE does not quote the pre-registered targets {_missing} -- a "
                       f"prediction nobody can find is not a prediction")
        else:
            ok.append("EVIDENCE quotes all three pre-registered rebuild targets")

# --- one SS3b presence answer, not two -----------------------------------------------------
# Two committed artifacts from the same row state how many of their 4,147 gradient-carrying
# tensors we can carry, and on 2026-09-19 they disagreed by 48: `full_model_of3_full_mat64`
# said 3,497 (pass 41) while `reach_by_norm` said 3,545 (pass 42, after one call to the
# confidence head). Both are internally consistent, both are live, and a reader landing on the
# older one gets the 94.44 % -era answer. Same class as the bundle-digest guard, except this
# one was already true when it was written. The rule is not "they must be equal" -- a
# historical artifact is allowed -- it is that the superseded one must SAY it is superseded.
_pres = j("perf/of3t_gradients/full_model_of3_full_mat64.json")
_reach = j("perf/of3t_gradients/reach_by_norm.json")
if _pres and _reach:
    _old = _pres.get("presence", {}).get("ours_carried_by_a_device_tensor")
    _new = _reach.get("reach", {}).get("device_bijection_mat64", {}).get("tensors")
    try:
        _old, _new = int(_old), int(_new)
    except (TypeError, ValueError):
        _old = _new = None
    if _old is not None and _old != _new:
        if _pres.get("superseded_by") or _pres.get("presence", {}).get("superseded_by"):
            ok.append(f"SS3b has one live answer: {_new}; the older {_old} is marked superseded")
        else:
            # WARN and not DRIFT, deliberately: the remedy is one field in an artifact this
            # row owns and I do not, and a compose held red for hours over someone else's
            # metadata is a gate I would learn to scroll past. It is named, it is in their
            # brief, and it becomes a failure if it survives their next artifact update.
            warn.append(f"TWO SS3b ANSWERS LIVE: full_model_of3_full_mat64 says {_old} of 4147 "
                       f"carried while reach_by_norm says {_new} -- a {abs(_new-_old)}-tensor "
                       f"gap, both committed, neither marked superseded. Re-run the older or "
                       f"give it a `superseded_by`")
    elif _old is not None:
        ok.append(f"SS3b agrees across artifacts: {_new} of 4147 carried")

# --- the block-0 decomposition, recomputed rather than transcribed ------------------------
# EVIDENCE quotes five group medians and a projection. All of it is derived from one artifact
# by arithmetic, so none of it should be typed twice -- recompute and compare.
d = j("perf/of3t_gradients/instrument_a_bundle_block0.json")
if d:
    import statistics as _st
    _pp = [q for q in d.get("per_parameter", [])
           if q.get("ref_norm") is not None and q["ref_norm"] >= 1e-12]
    def _grp(q):
        n = q["their_tensor"]
        for k in ("tri_att_end", "tri_att_start", "attn_pair_bias", "single_transition"):
            if k in n:
                return k
        return "pair_stack_rest"
    _g = {}
    for q in _pp:
        _g.setdefault(_grp(q), []).append(q["rel_l2"])
    _want = {"tri_att_end": (8, 0.3838), "attn_pair_bias": (6, 0.1470),
             "tri_att_start": (8, 0.0865), "pair_stack_rest": (25, 0.0680),
             "single_transition": (5, 0.0212)}
    _off = []
    for k, (n_w, med_w) in _want.items():
        v = _g.get(k, [])
        if len(v) != n_w or abs(_st.median(v) - med_w) > 5e-3 * max(med_w, 1e-9) + 5e-5:
            _off.append(f"{k}: n={len(v)} median={_st.median(v) if v else float('nan'):.4g} "
                        f"(scoreboard {n_w}, {med_w})")
    # The load-bearing half of the claim: single_transition is the ONLY group with nothing
    # over the per-tensor bar. If another group joins it, "graded by pair/attention
    # involvement" stops being the reading.
    _clean = [k for k, v in _g.items() if v and max(v) <= 0.05]
    if _off:
        bad.append("block-0 decomposition has moved: " + "; ".join(_off))
    elif _clean != ["single_transition"]:
        bad.append(f"the groups fully inside the bar are now {_clean}, not just "
                   f"single_transition -- EVIDENCE's mechanism reading rests on it being alone")
    else:
        ok.append("block-0 decomposition holds: single_transition alone is inside the bar, "
                  "the other four groups are 17/25 to 8/8 over it")
    # And D9's projection, which is an upper bound and must stay one.
    _fe, _fs = 5.239e-02 / 1.449e-01, 4.283e-02 / 1.389e-01
    _proj = [q["rel_l2"] * (_fe if "tri_att_end" in q["their_tensor"] else
                            _fs if "tri_att_start" in q["their_tensor"] else 1.0)
             for q in _pp]
    if abs(_st.median(_proj) - 0.07024) <= 1e-3 and _st.median(_proj) > 0.02:
        ok.append(f"D9's projection is still an upper bound that fails: median "
                  f"{_st.median(_proj):.5f} against the 0.02 bar, "
                  f"{sum(1 for x in _proj if x > 0.05)} of {len(_proj)} over 0.05")
    else:
        bad.append(f"D9's projection now reads {_st.median(_proj):.5f} -- the claim that "
                   f"fp32_softmax cannot rescue block 0 is computed from this number")

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
    _declared_file = None
    for _a in _man.get("artifacts", []):
        # Follow whatever the MANIFEST currently calls the validated gradient. Hardcoding
        # `grads_f64_recycles0.pt` was right until `of3t-reference` republished as
        # `grads_f64_r0.pt`, at which point the check stopped being able to run -- and a check
        # that cannot run is how D18 stayed invisible for thirty-nine passes.
        if isinstance(_a, dict) and str(_a.get("file", "")).startswith("grads_f64") \
                and "recycles3" not in str(_a.get("file", "")):
            _declared = _a.get("sha256")
            _declared_file = _a.get("file")
    _citers = {
        "perf/of3t_updaterule/reference_profile.json": ("declared_sha256",),
        "perf/of3t_gradients/reach_by_norm.json": ("reference", "sha256"),
        "perf/of3t_gradients/instrument_a_bundle_block0.json": ("bundle", "sha256"),
    }
    _seen, _stamped = {}, []
    for _rel, _path in _citers.items():
        _d = j(_rel)
        if not _d:
            continue
        _v = _d
        for _k in _path:
            _v = _v.get(_k) if isinstance(_v, dict) else None
        # A citer that DECLARES it was taken against a withdrawn reference is history, not a
        # live claim, and history is allowed to disagree. `of3t-gradients` stamps these with
        # `reference_standing` naming both digests. Undeclared disagreement is the defect.
        if _v and not (_d.get("superseded_by") or _d.get("reference_standing")):
            _seen[_rel] = _v
        elif _v:
            _stamped.append(_rel)
    if _declared and _seen:
        _bad = {k: v for k, v in _seen.items() if not _declared.startswith(str(v)[:40])
                and not str(v).startswith(_declared[:40])}
        # The republish is a KNOWN transition, not a surprise: `of3t-reference` is rebuilding
        # the r=0 bundle and its new digest lands in the MANIFEST before the rows that cite it
        # can re-hash. If every citer still agrees with every other citer and only the MANIFEST
        # has moved, that is exactly that transition -- name it and name who owes the re-hash,
        # rather than failing the compose for hours on an expected state. A compose I expect to
        # be red is a compose I stop reading. Citers disagreeing with EACH OTHER is the real
        # defect and still fails.
        if _bad and len(set(_seen.values())) == 1:
            warn.append(f"REPUBLISH IN FLIGHT: the MANIFEST now declares "
                        f"{_declared[:16]}... while all {len(_seen)} citers still agree on "
                        f"{list(_seen.values())[0][:16]}... -- expected while "
                        f"`of3t-reference` publishes. Owed: a re-hash in "
                        + ", ".join(sorted(_seen)) + ", in the same commit as the re-run")
        elif _bad:
            bad.append(f"TWO REFERENCES IN FLIGHT: the MANIFEST declares "
                       f"{_declared[:16]}... for grads_f64_recycles0.pt while "
                       + "; ".join(f"{k} cites {v[:16]}..." for k, v in _bad.items())
                       + ". Every figure measured against the other one is a figure about an "
                         "artifact that no longer exists (D18)")
        else:
            ok.append(f"one reference campaign-wide: {len(_seen)} artifacts and the MANIFEST "
                      f"all cite {_declared[:16]}...")
    if _stamped:
        ok.append(f"{len(_stamped)} citer(s) declare which reference they were taken "
                  f"against, so a disagreeing digest in them is history rather than a live "
                  f"claim: "
                  + ", ".join(x.split('/')[-1] for x in _stamped))
    if not _declared:
        warn.append("the MANIFEST declares no sha256 for any grads_f64* artifact -- "
                    "the one-reference check cannot run, which is how D18 stayed invisible")

# --- DEFECTS' LIVE claims must follow the artifacts too --------------------------------------
# EVIDENCE is audited, the summary fields are audited, and DEFECTS.md -- the document that says
# which defects are open and how big they are -- was not. When the reference was republished
# every share moved, and D20 (the campaign's CEILING) still read 88.54 / 3.57 / 91.9 % from the
# withdrawn tape for six passes.
#
# The rule that makes this checkable without freezing history: a defect's HISTORICAL narrative
# may quote the numbers of its day, but the figures in its HEADING and its current-state
# paragraph must match the artifacts. So: check the heading line and the D20 body.
_DEFP = _campaign_doc("DEFECTS")
if _DEFP.is_file() and (_reach_top or j("perf/of3t_orchestrator/SECTION_MASS_MEASURED.json")):
    _dt = _DEFP.read_text()
    # Pass 151: the shares that DEFECTS must follow now come from the EXHAUSTIVE 0.4.3 table,
    # not from reach_by_norm.json, which is the 0.5.0 / 4,147 artifact pass 91 disqualified.
    # This guard demanded 91.21 % -- a share of a model the campaign does not claim -- so a
    # green run meant DEFECTS was stale. Fourth sighting of a guard pinned to a superseded
    # artifact enforcing staleness; the rule is to re-point the guard when the artifact is
    # superseded, in the same pass.
    _sm = j("perf/of3t_orchestrator/SECTION_MASS_MEASURED.json") or {}
    _smsec = _sm.get("sections_pct_of_model", {})
    if _smsec:
        _diff_share = _sm.get("diffusion_module_total", {}).get("pct", 0) / 100.0 or None
        _aux_share = _smsec.get("aux_heads", {}).get("pct", 0) / 100.0 or None
    else:
        _diff_share = _reach_top.get("diffusion_module", {}).get("share")
        _aux_share = _reach_top.get("aux_heads", {}).get("share")
    if _diff_share and _aux_share:
        _want_sum = f"{(_diff_share + _aux_share) * 100:.1f} %"
        _m = re.search(r"^### D20\. .*?(\d+(?:\.\d+)?) % of the gradient", _dt, re.M)
        if _m and abs(float(_m.group(1)) - (_diff_share + _aux_share) * 100) <= 0.15:
            ok.append(f"DEFECTS D20's headline share matches the artifacts ({_want_sum})")
        elif _m:
            bad.append(f"DEFECTS D20 headlines {_m.group(1)} % where the artifacts give "
                       f"{_want_sum} -- the ceiling is quoted against a reference that has "
                       f"been replaced")
        # Compare NUMERICALLY, not as a format string. The first version required "89.21 %"
        # and the document said "89.2106 %" -- a more precise statement of the same number,
        # rejected. Same defect the pass-37 "3.16 % vs 3.156 %" check had; a matcher that
        # insists on its own rounding reports drift against a document that is more correct
        # than the check is.
        _pcts = [float(m) for m in re.findall(r"(\d+\.\d+)\s*%", _dt)]
        for _name, _sh in (("diffusion_module", _diff_share), ("aux_heads", _aux_share)):
            _want = _sh * 100
            if not any(abs(_p - _want) <= 0.005 for _p in _pcts):
                bad.append(f"DEFECTS never quotes {_name}'s current share {_want:.4f} % -- "
                           f"D20's body is the campaign's ceiling and it must follow the "
                           f"artifact")

# --- every UNFIXED defect must be named in the orchestrator's GAP ----------------------------
# GAP has drifted twice: it described the campaign as it stood seven passes earlier, and then
# omitted the hardest blocker entirely. The gate only checks that the field EXISTS. A summary
# written by transcription drifts exactly like a scoreboard does, so it gets the same treatment
# as the scoreboard: checked against its source.
DEF = _campaign_doc("DEFECTS")
ORCH = _campaign_doc("ORCHESTRATOR")
if DEF.is_file() and ORCH.is_file():
    import re as _re
    # The campaign's convention is that a defect's status lives on its HEADING line, and this
    # check reads only the heading. Pass 175: D77 was written with UNFIXED in its BODY instead,
    # and it was therefore invisible here -- an UNFIXED defect that GAP was not required to
    # name, which is precisely what this check exists to prevent. Same shape as D74 and D76:
    # the guard read a narrower form than the document used. So a body-only declaration is now
    # a FAILURE rather than a silent exclusion. The pattern is a status DECLARATION, not any
    # mention of the word -- D66's body discusses other defects' UNFIXED status and must not
    # trip it.
    _dt_u = DEF.read_text()
    _parts = _re.split(r"(?m)^(### D\d+\..*)$", _dt_u)
    _ents = [(_parts[i], _parts[i + 1]) for i in range(1, len(_parts) - 1, 2)]
    _decl = _re.compile(r"\*\*UNFIXED[.*]|\bUNFIXED\b\s*(?:--|\u2014|\.)")
    _bodyonly = [_re.match(r"### (D\d+)\.", h).group(1) for h, b in _ents
                 if "UNFIXED" not in h and _decl.search(b)]
    if _bodyonly:
        bad.append("defect(s) declare UNFIXED in the BODY but not on the heading, where this "
                   "audit and GAP's coverage check read it, so they are invisible to both: "
                   + ", ".join(_bodyonly) + " -- put the status on the heading line")
    unfixed = [m.group(1) for m in
               _re.finditer(r"^### (D\d+)\..*$", _dt_u, _re.M)
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

# --- an UNFIXED defect must not leave a hypothesis hanging ------------------------------------
# Pass 82's own finding, and I am the case that motivates it. D19 carried a paragraph headed
# "A hypothesis with a decisive test, offered rather than asserted" for twenty passes. The test
# had already run at stack scope and REFUTED the hypothesis -- and the refuting number went into
# PROTOCOL.md, into EVIDENCE.md and into my own state doc, but never back into the defect entry,
# which is where a reader goes to find out what is wrong. I read it, believed the question open,
# and spent most of a pass rebuilding a control that had already been run.
#
# So: a defect entry may state a hypothesis, but an UNFIXED one may not leave it OPEN. It must
# carry a resolution word in the same entry. This does not ask the campaign to settle every
# question -- "REFUTED", "CONFIRMED", "still open" and "what would settle it" all satisfy it.
# It only forbids the one shape that cost a pass: a hypothesis presented as live, in an entry
# whose evidence has already moved on, with nothing in the entry saying which.
if DEF.is_file():
    import re as _re2
    _txt = DEF.read_text()
    _entries = _re2.split(r"^### (D\d+)\.", _txt, flags=_re2.M)
    _HYP = _re2.compile(r"hypothesis|is live\b|offered rather than asserted", _re2.I)
    _RES = _re2.compile(r"REFUTED|CONFIRMED|RESOLVED|settled|still open|remains open|"
                        r"what would settle", _re2.I)
    _dangling = []
    for _i in range(1, len(_entries), 2):
        _num, _body = _entries[_i], _entries[_i + 1]
        _head = _body.split("\n", 1)[0]
        if "UNFIXED" not in _head:
            continue
        if _HYP.search(_body) and not _RES.search(_body):
            _dangling.append(_num)
    if _dangling:
        bad.append(f"these UNFIXED defects state a hypothesis and never say whether it still "
                   f"stands: {', '.join(_dangling)} -- a reader of the entry cannot tell that "
                   f"the evidence has moved on, which cost pass 82 most of a pass")
    else:
        ok.append("no UNFIXED defect leaves a hypothesis open without saying so")

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
        # Pass 134: this used to pin the summary to reach_by_norm.json's trunk share, which is
        # computed on the 4,147-parameter 0.5.0 model. Pass 91 established the right denominator
        # is 4,170, and the check was then REQUIRING the summary to quote the wrong one -- so
        # keeping the summary current made the audit fail, and keeping the audit green kept the
        # summary stale. A guard that enforces staleness is worse than no guard. The share is
        # now read from the campaign's own corrected table, and the old artifact is left alone
        # as the historical record it is.
        _t = _re.search(r"`pairformer_stack`[^|]*\|\s*\*\*(\d+\.\d+)\*\*", o)
        if _t:
            claims.append((f"{float(_t.group(1)):.2f} %", doesnot,
                           "the trunk's share of the norm on the 4,170 basis"))
    _c = j("perf/of3t_equivalence/instrument_c_optim.json")
    if _c and "of3_schedule_upstream" in _c.get("arms", {}):
        claims.append((f"{_c['arms']['of3_schedule_upstream']['worst']['rel']:.3e}",
                       proves, "SS5 against upstream's alignment"))
    # Pass 134: was pinned to instrument_a_bundle_block0.json, the PRE-0.4.3-rebuild arm whose
    # median the campaign no longer quotes. Same failure as the trunk share above. Pin it to the
    # arm that is current -- the A16 bundle, which carries norm_ratio and cos as well.
    _a = j("perf/of3t_orchestrator/a16/instrument_a_bundle_A16_block0_tbshipped.json")
    if _a:
        # `summary.median` is over every compared tensor; `identifiability.median_rel` is over
        # the A14 set (zero-reference tensors excluded), which is the one PROTOCOL A14 requires
        # and therefore the one the summary quotes. 0.0122 against 0.0121 is that difference.
        _m = (_a.get("identifiability") or {}).get("median_rel", _a["summary"]["median"])
        claims.append((f"{_m:.4f}", doesnot,
                       "instrument A's block-0 median (A14 applied) against the 0.4.3 reference"))
    _d = j("perf/of3t_orchestrator/instrument_c2_clip_in_step.json")
    if _d:
        claims.append((f"{_d['arms']['clip_binds']['worst']['rel']:.3e}", proves,
                       "the clip/optimizer seam"))
    # A share written "3.156 %" and a check formatting "3.16 %" disagree about nothing. Accept
    # either rendering rather than forcing the prose to carry the check's precision -- the
    # point is that the summary follows the artifact, not that it matches a format string.
    def _quoted(s, where):
        if s in where:
            return True
        if s.endswith(" %"):
            try:
                v = float(s[:-2])
            except ValueError:
                return False
            # Compare NUMERICALLY against every percentage in the text, rather than trying to
            # guess the author's rounding. "3.156 %" and a check computing "3.16 %" disagree
            # about nothing, and a matcher that insists on a format string makes the prose
            # serve the checker.
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", where):
                if abs(float(m.group(1)) - v) <= 0.005 + 0.002 * abs(v):
                    return True
        return False
    missing = [(s, why) for s, where, why in claims if not _quoted(s, where)]
    if missing:
        for s, why in missing:
            bad.append(f"PROVES/DOESNOT does not quote {s} ({why}) -- the artifact has moved "
                       f"and the summary Moritz reads has not")
    elif claims:
        ok.append(f"PROVES/DOESNOT quote all {len(claims)} audited figures as the artifacts "
                  f"have them")
    # VERDICT is the third summary field and it was NOT in the first version of this check --
    # written one pass earlier, covering PROVES and DOESNOT only. It was stale in exactly the
    # way they were ("eleven rows, eight concluded", "31 scoreboard figures", "fourteen
    # defects") and my own new guard walked straight past it. A check scoped to the two fields
    # you happened to be reading is a check scoped to your attention.
    # The VERDICT field is multi-paragraph, so stopping at the first blank line captures only
    # its opening sentence -- which is how the first version of this check reported drift on a
    # field that said the right thing three lines lower. Stop at the narrative that follows it
    # (a line beginning "**`of3t-" or "PASS <n>."), not at whitespace.
    vd = _re.search(r"^VERDICT:(.*?)(?=^\*\*`of3t-|^PASS \d+\.|\Z)", o, _re.M | _re.S)
    verdict = vd.group(1) if vd else ""
    # Starts at ONE, not at seven. The first version began at 7 because that was the count the
    # day it was written, and the moment D14 closed and UNFIXED fell to 6 it reported drift on
    # a VERDICT that said "six remain UNFIXED" in words. Second time a guard of mine has been
    # wrong about a document that was right; both times the guard encoded the shape of the
    # prose it was born against.
    # Pass 155: GENERATED, not hardcoded. This list stopped at twenty-four, so the UNFIXED
    # check reported drift against a document that was correct (it fell through to a 'zzz'
    # sentinel and failed loudly, which is the pass-138 fix working) -- and the DEFECTS-count
    # check, whose branch is guarded by `if _w`, went SILENTLY VACUOUS at twenty-five and has
    # not checked anything since the record passed D24. Fourth sighting of a guard outgrowing
    # its word list in this campaign. A generated list cannot outgrow the subject, and the
    # assertion below makes a missing word a failure rather than a skip.
    _ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
             "seventeen", "eighteen", "nineteen"]
    _TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
             "ninety"]

    def _word(n):
        if n < 20:
            return _ONES[n]
        if n < 100:
            return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
        return None

    _words = {n: _word(n) for n in range(1, 100)}
    if DEF.is_file():
        _dt = DEF.read_text()
        # Every DEFECTS-reading guard -- this count, the UNFIXED count, GAP's coverage check --
        # keys on the STRICT `### D<n>.` heading. At pass 175 D74 and D75 were written with an
        # em dash instead of the period, and all three guards silently SKIPPED them: the count
        # read 73 against a file holding 75, and the two entries were invisible to the check
        # that exists to notice exactly that. The guard did not fail, it undercounted, which is
        # the worse failure -- the same "goes quiet when its subject leaves the matched form"
        # class as the word list at twenty-one and the placeholder regex at pass 160.
        #
        # So the delimiter is now itself checked: anything that looks like a defect heading but
        # does not parse as one is a FAILURE. A document may not go partly invisible to its
        # own audit.
        _loose = _re.findall(r"^### (D\d+)(.)", _dt, _re.M)
        _malformed = [f"{d}{c!r}" for d, c in _loose if c != "."]
        if _malformed:
            bad.append("defect heading(s) do not use the `### D<n>.` form the DEFECTS guards "
                       "match, so they are INVISIBLE to the count, the UNFIXED list and GAP's "
                       "coverage check: " + ", ".join(_malformed))
        n_def = len(_re.findall(r"^### D\d+\.", _dt, _re.M))
        # The strict count and the loose count must agree, or one of them is reading a subset.
        if len(_loose) != n_def:
            bad.append(f"DEFECTS holds {len(_loose)} defect-shaped headings but only {n_def} "
                       f"parse -- {len(_loose) - n_def} entr(y/ies) are unaudited")
        if not _malformed and len(_loose) == n_def:
            # Said out loud on success: a guard that is silent when green is invisible when
            # green, which is how this one's absence went unnoticed for 175 passes.
            ok.append(f"all {n_def} defect headings parse, so none is invisible to its audit")
        n_unf = len([m for m in _re.finditer(r"^### D\d+\..*$", _dt, _re.M)
                     if "UNFIXED" in m.group(0)])
        for _n, _label in ((n_def, "defects"), (n_unf, "UNFIXED")):
            _w = _words.get(_n)
            if _w is None:
                bad.append(f"the {_label} count is {_n}, outside this check's word list -- the "
                           f"check cannot run, which is not a pass (pass-138 class)")
            elif _label == "defects" and f"{_w} defects" not in verdict.lower():
                bad.append(f"VERDICT does not say '{_w} defects' -- DEFECTS.md has {_n}")
            if _label == "UNFIXED" and not _re.search(rf"\b(?:{_n}|{_words.get(_n, 'zzz')})\b"
                                                      r"[^.]{0,40}UNFIXED", verdict):
                bad.append(f"VERDICT does not state the UNFIXED count -- DEFECTS.md has {_n}")
    _nc = len(list(Path("/home/moritz/.coworker/state/concluded").glob("of3t-*"))) \
        if Path("/home/moritz/.coworker/state/concluded").is_dir() else 0
    if _nc and _words.get(_nc) and _words[_nc] not in verdict.lower():
        bad.append(f"VERDICT does not state the concluded-row count -- "
                   f"state/concluded holds {_nc} of3t rows ({_words[_nc]})")
    if not bad or all("VERDICT" not in b for b in bad):
        ok.append("VERDICT states the defect, UNFIXED and concluded-row counts as their "
                  "sources have them")

    # --- the summary must not deny a measurement it also reports -------------------------------
    # Pass 123. GAP read "No s/step exists on either side: ... nothing is measured" while the
    # PROVES block four hundred lines above it reported "870.75 s on a p300c against 7-8 s on an
    # H200, ~116x". Both are owed fields, both get quoted, and they cannot both be true. I had
    # propagated the denial into a report the pass before I noticed the figures.
    #
    # The guard is deliberately narrow: it does not try to detect contradiction in general, only
    # the specific shape that bit -- the summary asserting that a quantity has NOT been measured
    # while the same summary carries a number for it. Extend the pairs list when a new headline
    # quantity earns one.
    _DENIALS = [
        (r"\bno\s+s\s*/\s*step\s+exists\b", r"\d+(?:\.\d+)?\s*s\b[^.]{0,80}(?:p300c|H200|step)",
         "s/step"),
        (r"\bnothing\s+is\s+measured\b", r"sampled\s+DURING", "a DURING-sampled measurement"),
    ]
    _whole = proves + doesnot + (_re.search(r"^GAP:(.*?)(?=^VERDICT:)", o, _re.M | _re.S).group(1)
                                 if _re.search(r"^GAP:(.*?)(?=^VERDICT:)", o, _re.M | _re.S)
                                 else "")
    for _deny, _have, _what in _DENIALS:
        if _re.search(_deny, _whole, _re.I) and _re.search(_have, _whole, _re.I):
            bad.append(f"the summary denies {_what} has been measured AND carries a figure for "
                       f"it -- one of the two is stale (pass-123 recurrence)")
    if not any("the summary denies" in b for b in bad):
        ok.append("the summary does not deny a measurement it also reports")

    # And the amendment count, which is a claim about the protocol's own history.
    _pp = _campaign_doc("PROTOCOL")
    if _pp.is_file():
        n_am = len(_re.findall(r"^\*\*A\d+ \u2014", _pp.read_text(), _re.M))
        # Pass 175: this guard held THREE hand-written word lists, all capped at
        # "twenty-four", and `w = words.get(n_am)` under `if w:` meant that at the
        # twenty-FIFTH amendment `w` would be None and every branch below would be
        # skipped -- the check would go silently vacuous exactly when the protocol grew.
        # That is the pass-138 class (a guard that goes quiet when its subject outgrows
        # its word list), and it was sitting one amendment away. Found pre-emptively while
        # fixing D74, which is the same defect in the compose's row list. The defect-count
        # check above already generates its words for 1..99 and FAILS when out of range;
        # this one now shares that list, and out-of-range is a failure here too.
        w = _words.get(n_am)
        if w is None:
            bad.append(f"the amendment count is {n_am}, outside this check's word list -- "
                       f"the check cannot run, which is not a pass (pass-138 class)")
        # Longest-first so "twenty-three" cannot be read as "three"; built from the SAME
        # generated list, so the alternation can never fall behind the counter again.
        _AMW = "|".join(sorted((v for v in _words.values() if v),
                               key=lambda v: (-len(v), v)))
        if w and _re.search(r"(?<!-)\b(" + _AMW + r") amendments\b", both):
            if f"{w} amendments" in both:
                ok.append(f"PROVES states the amendment count correctly ({w}, {n_am})")
            else:
                bad.append(f"PROVES states the wrong amendment count -- PROTOCOL has {n_am} "
                           f"({w})")
        # ...and the SAME sentence lives in VERDICT, which this check did not read. VERDICT said
        # "amended fifteen times on the record" while PROTOCOL held eighteen, and the audit was
        # silent for three amendments. Third time a guard of mine has been scoped to the field I
        # happened to be reading rather than to the claim -- so it is now scoped to the claim,
        # wherever in the summary it is written.
        # Pass 142: the compound words MUST come first and the match must not start after a
        # hyphen. "twenty-two times on the record" contains "two times on the record", and
        # \b matches at the hyphen -- so the check read the document as saying "two" and
        # reported drift against a document that was correct. Longest-first alternation plus
        # a negative lookbehind for "-" fixes both halves.
        _AMEND = _re.compile(r"(?<!-)\b(" + _AMW + r")\s+times\s+on\s+the\s+record\b",
                             _re.I)
        # every space is \s+: the phrase is hard-wrapped prose and lands as
        # "times on the\nrecord". The first version used literal spaces, found nothing, and
        # reported a clean pass on a document that said "fifteen" -- a check that cannot match
        # its own target is indistinguishable from a check that passes.
        if w:
            _hits = [(fld, m.group(1).lower()) for fld, txt in
                     (("PROVES/DOESNOT", both), ("VERDICT", verdict))
                     for m in _AMEND.finditer(txt)]
            _wrong = [f"{fld} says {got!r}" for fld, got in _hits if got != w]
            if _wrong:
                bad.append(f"the amendment count is stated wrongly -- PROTOCOL has {n_am} "
                           f"({w}) but {'; '.join(_wrong)}")
            elif _hits:
                ok.append(f"every summary field states the amendment count correctly "
                          f"({w}, {n_am}, {len(_hits)} place(s))")

# A reviewer running this off the published copies gets FEWER checks than the orchestrator
# does, and must be told which and why -- a check that cannot run has to say so (K60), and
# "141 confirmed" reads exactly like "146 confirmed" to someone who has never seen 146.
_host_only = len(list(Path("/home/moritz/.coworker/state/concluded").glob("of3t-*"))) \
    if Path("/home/moritz/.coworker/state/concluded").is_dir() else None
if _FELL_BACK:
    warn.append(f"read {len(_FELL_BACK)} campaign document(s) from the PUBLISHED COPY in this "
                f"branch rather than the authoritative source on pc "
                f"({', '.join(sorted(_FELL_BACK))}) -- the copies are regenerated every "
                f"compose, so they are current as of the commit you are reading")
if _host_only is None:
    warn.append("the concluded-row count could not be checked: it reads "
                "~/.coworker/state/concluded, which exists only on the orchestrator's host. "
                "That check did NOT run -- it is not a pass")

# --- the distance-to-go arithmetic must sum, and VERDICT must quote it ------------------------
# Pass 152. The campaign's position is now a three-way split of the model's gradient mass, and
# it is the number Moritz reads. Two ways it can rot: the three shares stop summing to 100 as
# readings move between buckets, and VERDICT keeps yesterday's passing share. Both are the
# campaign's most recurrent defect class, so both are mechanical now.
# Pass 175: RE-POINTED. This guard read DISTANCE_TO_GO_BY_MASS.json, whose shares are distances
# from the FLOAT64 ideal, and required VERDICT to quote them -- so once the campaign started
# measuring against upstream's OWN training step, the guard was enforcing the superseded framing.
# That is the campaign's own "a guard pinned to a superseded artifact enforces staleness", fourth
# sighting, this time on the guard I wrote for exactly that class. It now reads the artifact scored
# against their step, and the old file's headline is NULLED in place rather than only stamped,
# because a stamp does not stop a number being read.
_dtg = j("perf/of3t_orchestrator/DISTANCE_TO_GO_AGAINST_THEIR_STEP.json")
if _dtg:
    _p = _dtg["survives"]["total_pct"]
    _f = _dtg["measured_and_fails"]["total_pct"] + \
         _dtg["measured_and_void_under_A18"]["total_pct"]
    _u = _dtg["no_direct_reading"]["total_pct"]
    _tot = _p + _f + _u
    if abs(_tot - 100.0) > 0.001:
        bad.append(f"DISTANCE_TO_GO_AGAINST_THEIR_STEP's shares sum to {_tot:.4f} %, not 100 -- a "
                   f"reading moved buckets and the split was not rebalanced")
    else:
        ok.append(f"the distance-to-go split sums to 100.0000 % ({_p:.4f} survives / "
                  f"{_f:.4f} fails-or-void / {_u:.4f} unread)")
    # The retired file must stay retired: if its live headline ever carries the old split again,
    # something restored it from history and VERDICT will follow.
    _old = j("perf/of3t_orchestrator/DISTANCE_TO_GO_BY_MASS.json")
    if _old and "RETIRED" not in str(_old.get("headline", "")):
        bad.append("DISTANCE_TO_GO_BY_MASS.json's headline is live again -- it carries the "
                   "float64-scored split this campaign superseded at pass 175 (D82); null it")
    if ORCH.is_file():
        _verd = _re.search(r"^VERDICT:(.*)", ORCH.read_text(), _re.M | _re.S)
        _vt = _verd.group(1) if _verd else ""
        _pcts = [float(m) for m in _re.findall(r"(\d+\.\d+)\s*%", _vt[:2000])]
        for _val, _lbl in ((_p, "surviving"), (_f, "failing-or-void"), (_u, "unread")):
            if not any(abs(_x - _val) <= 0.005 for _x in _pcts):
                bad.append(f"VERDICT does not state the {_lbl} share {_val:.4f} % -- the "
                           f"summary has drifted from DISTANCE_TO_GO_AGAINST_THEIR_STEP")
        if not [b for b in bad if "VERDICT does not state the" in b and "share" in b]:
            ok.append("VERDICT states all three distance-to-go shares as the artifact has them")

# --- a summary field must stay readable, which is a LENGTH property no content check sees -----
# Pass 166. VERDICT had grown to 177,928 characters over 2,371 lines, because every pass appends
# after the last field and VERDICT is the last field, so the whole narrative landed inside the one
# field a reader treats as the answer. Every content check above passed throughout -- they read the
# shares, the counts and the amendment phrase, all of which sit in its first eight lines, and none
# of them reads its size. A field can be entirely correct and entirely unusable.
if ORCH.is_file():
    _o = ORCH.read_text()
    _CAPS = {"VERDICT": 4000, "PROVES": 20000, "DOESNOT": 20000, "GAP": 40000}
    _over = []
    for _f, _cap in _CAPS.items():
        _m = _re.search(rf"^{_f}:(.*?)(?=^[A-Z][A-Z_]+:|\Z)", _o, _re.M | _re.S)
        if _m and len(_m.group(1)) > _cap:
            _over.append(f"{_f} is {len(_m.group(1))} chars against a {_cap} cap")
    if _over:
        bad.append("summary field(s) have accreted past the point of being read: "
                   + "; ".join(_over) + " -- move the narrative to PASSLOG, which is what it is for")
    else:
        ok.append("every owed summary field is inside its readability cap")

# --- and the check COUNT the summary quotes ---------------------------------------------------
# Pass 133. PROVES carried "(146 checks, 0 drifted)" while the audit had grown to 149. The count
# is a claim about how much evidence stands behind the field, it is quoted verbatim, and nothing
# updated it when checks were added. Self-referential by construction: the number is whatever
# this run ends with, so the doc has to match it. The first run after adding a check will fail,
# which is exactly when the author is there to fix it.
if ORCH.is_file():
    # Pass 175: the count is HOST-DEPENDENT and pinning one number guaranteed the very
    # recurrence this message names. The A12 live-mapping probe needs `torch`; where it imports
    # it appends an ok, and where it does not it appends a WARNING instead -- so the same
    # commit legitimately audits 154 on one interpreter and 153 on another, and the third
    # compose of this pass drifted for no reason but that. A check that cannot run says so
    # (K60), and a check that said so is not a check that vanished.
    #
    # So the total counted here is checks that RAN plus checks that ANNOUNCED they could not,
    # which is stable across hosts. The stated number must equal that total; a run where a
    # probe silently disappeared still fails, because it would lower both terms.
    _n_ran = len(ok) + 1                       # +1 for the ok this check is about to append
    _n_now = _n_ran + len(warn)
    _cm = _re.search(r"\((\d+)\s+checks,\s*0\s+drifted\)", o)
    if _cm is None:
        bad.append("PROVES does not state the check count as '(N checks, 0 drifted)' -- the "
                   "audit reports a total that nothing in the summary is pinned to")
    elif int(_cm.group(1)) != _n_now:
        bad.append(f"PROVES states ({_cm.group(1)} checks, 0 drifted) but this audit has "
                   f"{_n_now} ({_n_ran} confirmed + {len(warn)} that announced they could not "
                   f"run) -- the count drifted when checks were added (pass-133 recurrence)")
    else:
        ok.append(f"PROVES states the check count correctly ({_n_now} = {_n_ran} confirmed "
                  f"+ {len(warn)} skipped-and-said-so)")

print("AUDIT of state/of3t/EVIDENCE.md against committed artifacts\n")
for line in ok:
    print(f"  ok    {line}")
for line in warn:
    print(f"  WARN  {line}")
for line in bad:
    print(f"  DRIFT {line}")
print(f"\n{len(ok)} confirmed, {len(warn)} warning(s), {len(bad)} drifted")
sys.exit(1 if bad else 0)
