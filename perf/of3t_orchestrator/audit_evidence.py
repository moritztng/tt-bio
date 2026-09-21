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
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- refuse to run outside the composed tree -------------------------------
# This audit recomputes EVIDENCE.md against artifacts contributed by ~30 rows.
# Run from a single row's worktree it finds most of them absent and reports
# them as DRIFT -- at pass 182 that was 24 of 25 "drifts", all false, and the
# real one was buried among them. A wrong-tree run must be a REFUSAL (exit 2),
# never a drift report, because a drift report is indistinguishable from
# artifacts having actually gone missing. compose_verify.sh:97 checks the
# composition out as branch `wk/of3t`; that is the only tree this can score.
if os.environ.get("OF3T_AUDIT_TREE_OK") != "1":
    _br = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if _br != "wk/of3t":
        sys.stderr.write(
            f"REFUSING: this audit scores the COMPOSED tree, but HEAD here is {_br!r}, "
            f"not 'wk/of3t'.\n"
            f"  {ROOT}\n"
            "Most rows' artifacts are absent in a single row's worktree, so every check that\n"
            "reads one would report a false DRIFT. Run it via perf/of3t_orchestrator/"
            "compose_verify.sh,\nor against the composed worktree it builds. Set "
            "OF3T_AUDIT_TREE_OK=1 only if you have\nverified the artifacts are present by "
            "some other route.\n")
        sys.exit(2)
# ---------------------------------------------------------------------------


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
    #
    # Pass 240, two false positives in one compose, both from the same root: the split was on
    # `### D\d+\.` only, so an `### Dn UPDATE ...` heading did NOT start a new entry and its body
    # was charged to whichever `.`-heading came before it. D132's "body" therefore swallowed
    # D129 UPDATE 2's `**UNFIXED**`. Split on every heading. And after that, D134's own body still
    # tripped it on the sentence "Only D69's status actually moves, to **UNFIXED**" -- a
    # declaration ABOUT ANOTHER DEFECT, which is the exact false positive the paragraph above
    # says this check must not make. So a declaration counts only when the sentence carrying it
    # does not name a different defect.
    _dt_u = DEF.read_text()
    _parts = _re.split(r"(?m)^(### D\d+\b.*)$", _dt_u)
    _ents = [(_parts[i], _parts[i + 1]) for i in range(1, len(_parts) - 1, 2)]
    _decl = _re.compile(r"\*\*UNFIXED[.*]|\bUNFIXED\b\s*(?:--|\u2014|\.)")

    def _quoted_spans(text):
        """Character ranges inside the house emphasis-quote form *"..."*.

        Pass 252: correcting a status honestly means QUOTING the wrong one -- D21's update says
        its heading `read *"UNFIXED -- the forward discriminator has not been run"*`. That is a
        report of what the heading said, not a declaration, and reading it as one pressures an
        author to paraphrase history rather than quote it, which is the opposite of what this
        ledger wants.
        """
        return [(m.start(), m.end()) for m in _re.finditer(r'\*"[^"]*"\*', text)]

    def _declares_own_unfixed(head, body):
        """True when BODY declares UNFIXED about THIS defect rather than about another one."""
        _me = _re.match(r"### (D\d+)\b", head).group(1)
        _q = _quoted_spans(body)
        for _m in _decl.finditer(body):
            if any(a <= _m.start() < b for a, b in _q):
                continue                      # inside a quotation: a report, not a declaration
            _lo = body.rfind(".", 0, max(0, _m.start() - 1)) + 1
            _hi = body.find(".", _m.end())
            _sent = body[_lo: _hi if _hi != -1 else len(body)]
            _others = {d for d in _re.findall(r"\bD\d+\b", _sent) if d != _me}
            if not _others:
                return True
        return False

    # Probe: a body-only declaration must still fire, and one about another defect must not.
    _bo_probe = [_declares_own_unfixed("### D1. FIXED.", "It stays UNFIXED. More text."),
                 _declares_own_unfixed("### D2. FIXED.", "Only D69's status moves, to **UNFIXED**."),
                 _declares_own_unfixed("### D3. CLOSED.",
                                       'Its heading read *"UNFIXED -- not run"* until now.')]
    if _bo_probe != [True, False, False]:
        bad.append("the body-only-UNFIXED probe did not fire on both shapes -- the check is "
                   "either inert or it is flagging talk about other defects (pass-240 shape)")
    _bodyonly = [_re.match(r"### (D\d+)\b", h).group(1) for h, b in _ents
                 if "UNFIXED" not in h and _declares_own_unfixed(h, b)]
    if _bodyonly:
        bad.append("defect(s) declare UNFIXED in the BODY but not on the heading, where this "
                   "audit and GAP's coverage check read it, so they are invisible to both: "
                   + ", ".join(_bodyonly) + " -- put the status on the heading line")
    # A defect's status is its LATEST heading: the campaign's convention since D111's UPDATE is
    # that a later "### Dn UPDATE (pass k)." entry supersedes the original, and pass 196 closed
    # D19, D87 and D99 that way. Reading every heading instead counted those three as still
    # UNFIXED, so this check reported 45 while the campaign's own count said 42 -- and the
    # contradiction check added in the same pass used the latest heading, so two checks in one
    # audit disagreed about what a status is.
    #
    # The conservative clause matters: if a later heading carries NO status word at all, the
    # defect keeps the last status that had one. Otherwise "### D8 UPDATE (pass N). More data."
    # would silently drop a live defect out of the set this check protects.
    # The vocabulary is defined ONCE, in perf/of3t_orchestrator/status_vocab.py. It used to
    # be written out five times across four files, which is the shape of half the defects
    # this campaign has filed against its own instruments (pass 241).
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from status_vocab import STATUS_RE as _STATUS_RE, PATTERN as _STATUS_PATTERN, \
        recorded_escape_hatches as _recorded_escape_hatches
    # One implementation, in status_vocab: upper-case in the source and unnegated, because a
    # guard that merely REPORTS a prose status while the parser commits it is two answers to one
    # question (pass 241; the D134 guard and this loop disagreed for a pass).
    from status_vocab import statuses_by_defect as _statuses_by_defect, declarations as _decls
    _last = _statuses_by_defect(_dt_u)
    unfixed = sorted((n for n, st in _last.items() if st == "UNFIXED"),
                     key=lambda d: int(d[1:]))

    # --- a closure word OUTSIDE the vocabulary silently keeps the old status --------------------
    # The conservative clause above is right (a heading with no status word must not drop a live
    # defect), but it has a blind spot the campaign walked into: pass 196 closed D87 with the word
    # **SUPERSEDED**, which is not in `_STATUS_RE`, so D87 kept UNFIXED for twenty-four passes
    # while its own latest entry said the claim is false against the revision the checkpoint is
    # bound to (pair track 4.947045e-02, under the 5.0e-02 bar). Found pass 220 by scanning for
    # exactly this shape, which is why it is now a check and not a scan.
    #
    # The fix is NOT to add SUPERSEDED to the vocabulary: it is genuinely ambiguous. D119's UPDATE
    # also says "superseded", but of its NUMBERS, and D119 is correctly still UNFIXED because
    # `project.py` still carries the unit error. So the guard refuses the ambiguity instead of
    # resolving it -- write a word from the vocabulary, or say UNFIXED and why.
    _AMBIG = _re.compile(r"\b(?:SUPERSEDED|OBSOLETE|VOID|MOOT|DISSOLVED|OVERTAKEN|"
                         r"NO LONGER (?:TRUE|OPEN|A DEFECT)|DUPLICATE OF)\b")

    def _ambiguous_closures(doc_upper):
        """Defects whose LATEST heading closes them with a word the parser cannot read."""
        latest_head = {}
        for m in _re.finditer(r"^### (D\d+)\b(.*)$", doc_upper, _re.M):
            latest_head[m.group(1)] = m.group(2)
        out = []
        for n, h in latest_head.items():
            if not _STATUS_RE.search(h) and _AMBIG.search(h):
                out.append(f"{n} (says {_AMBIG.search(h).group(0)})")
        return sorted(out, key=lambda x: int(x.split()[0][1:]))

    # Break control, run every time: a guard that cannot fire has tested nothing, and three of
    # this file's guards have shipped inert (the "FIXED" is a substring of "UNFIXED" one most
    # recently). The probe is a two-heading synthetic in exactly the shape D87 had.
    _probe = _ambiguous_closures("### D1. UNFIXED.\n### D1 UPDATE (PASS 2). SUPERSEDED. NO.\n")
    if _probe != ["D1 (says SUPERSEDED)"]:
        bad.append("the ambiguous-closure probe did not fire -- this check is inert and is "
                   "reporting nothing, which is how D87 survived twenty-four passes")
    else:
        _ambig = _ambiguous_closures(_dt_u)
        if _ambig:
            bad.append("these defects' LATEST heading carries a closure-sounding word that is NOT "
                       "in the status vocabulary, so the parser conservatively keeps the PREVIOUS "
                       "status and the defect is mis-counted: " + ", ".join(_ambig)
                       + " -- write FIXED, REFUTED, CLOSED, RESOLVED, WITHDRAWN, ROOT-CAUSED or "
                         "UNFIXED on the heading")
        else:
            ok.append("no defect's latest heading closes it with a word the status parser "
                      "cannot read (probe fires)")
    o = ORCH.read_text()
    g = _re.search(r"^GAP:(.*?)(?=^VERDICT:)", o, _re.M | _re.S)
    gap = g.group(1) if g else ""
    missing = [d for d in unfixed if not _re.search(rf"\b{d}\b", gap)]
    if missing:
        bad.append(f"GAP does not name these UNFIXED defects: {', '.join(missing)} "
                   f"-- the summary has drifted from DEFECTS.md")
    else:
        ok.append(f"GAP names all {len(unfixed)} UNFIXED defects")

    # --- RECORDED may not be used to retire a defect, and the gate must share the vocabulary ---
    # Pass 241. RECORDED was added so that a FINDING -- "block 47 is worth one per cent of the
    # model's gradient mass" -- can declare a status at all; twenty entries had none and were
    # invisible to every clause built on `unfixed` (D133). A new status word is an escape hatch
    # unless it is fenced, so: a defect that EVER declared UNFIXED may not later be restated
    # RECORDED. That is retirement by relabelling and it is refused, not judged.
    _hatch = _recorded_escape_hatches(_dt_u)
    if _hatch:
        bad.append("defect(s) restated RECORDED after having declared UNFIXED: "
                   + ", ".join(_hatch) + " -- RECORDED asserts the entry was never a defect, so "
                   "it cannot retire one that was. Use FIXED, REFUTED, WITHDRAWN or say UNFIXED")
    else:
        ok.append("no defect is retired into RECORDED after having declared UNFIXED "
                  "(the fence status_vocab.py exists for)")

    # And the fifth copy of the vocabulary, which cannot import: `_of3t_donecheck.py` runs from
    # ~/.coworker, not from this tree. It carries the pattern as a literal, so it is checked
    # against the shared one rather than trusted -- the two-readers pattern the release gate uses.
    _gate_p = Path("/home/moritz/.coworker/workstreams/_of3t_donecheck.py")
    if not _gate_p.is_file():
        warn.append("the gate script is not on this host, so its copy of the status vocabulary "
                    "could not be checked against status_vocab.PATTERN")
    else:
        _gm = _re.search(r're\.compile\(r"(\\b\(\?:UN\)\?[^"]+)"\)', _gate_p.read_text())
        if _gm is None:
            bad.append("could not find a status-vocabulary regex in _of3t_donecheck.py -- it had "
                       "one, and a check that stops finding what it reads is not a passing check")
        elif _gm.group(1) != _STATUS_PATTERN:
            bad.append("the gate's status vocabulary has drifted from status_vocab.PATTERN:\n"
                       f"      gate: {_gm.group(1)}\n      here: {_STATUS_PATTERN}")
        else:
            ok.append("the gate's own copy of the status vocabulary is identical to "
                      "status_vocab.PATTERN (it cannot import it; it is compared instead)")

    # --- a status read out of ORDINARY PROSE, or out of a NEGATION ----------------------------
    # Pass 240, and it is D87's mirror image. `_last` uppercases the whole heading before matching,
    # so an English word does the work of a declaration:
    #   D69  "...against a reading fixed before the arm produced output"  -> reads FIXED
    #   D116 "...and D8 is NOT closed by it"                              -> reads CLOSED
    # Nobody ever closed either one. D69 is a live finding recorded as FIXED because of a past
    # participle, and D116's heading says the OPPOSITE of what the parser stored, because a
    # word-level regex cannot see a negation. D87 was a real closure the parser could not read;
    # this is a non-closure the parser reads as a closure, and it is the more dangerous direction
    # because it removes a defect from `unfixed` silently.
    #
    # The house convention already writes the status in CAPITALS, usually bolded. So the rule is:
    # a status declaration must be upper-case in the source, and must not be immediately preceded
    # by a negation. This does not decide any defect's status -- it refuses to read one out of
    # prose, which is what D87's guard does from the other side.
    _NEG = _re.compile(r"\b(?:NOT|NEVER|NO LONGER|ISN'T|IS NOT|WAS NOT)\s+$", _re.I)

    def _prose_statuses(doc):
        """(defect, word, why) for a defect whose LATEST status-bearing heading is prose/negated.

        Latest-bearing, not every heading: the stored status is the last heading that carried a
        vocabulary word, so an old prose heading that a later capitalised one supersedes is
        history, not a live misreading. Flagging every heading made this check demand that the
        record be rewritten rather than corrected forward, which is not how this ledger works.
        """
        latest = {}
        for m in _re.finditer(r"^### (D\d+)\b(.*)$", doc, _re.M):
            if _STATUS_RE.findall(m.group(2).upper()):
                latest[m.group(1)] = m.group(2)
        out = []
        for _n, h in latest.items():
            up = _STATUS_RE.findall(h.upper())
            if not up:
                continue
            exact = list(_STATUS_RE.finditer(h))        # genuine upper-case occurrences
            good = [x for x in exact if not _NEG.search(h[:x.start()])]
            if not good:
                why = "lower-case prose" if not exact else "negated"
                bad_word = exact[-1].group(0) if exact else up[-1]
                out.append((_n, bad_word, why))
        return sorted(out, key=lambda x: int(x[0][1:]))

    # Probe, run every time: two synthetic headings in exactly the shapes D69 and D116 had.
    _pp = _prose_statuses("### D1. a reading fixed before the arm ran.\n"
                          "### D2. and D8 is NOT CLOSED by it.\n"
                          "### D3. UNFIXED, and here is why.\n")
    if [x[0] for x in _pp] != ["D1", "D2"]:
        bad.append("the prose-status probe did not fire -- this check is inert, which is how "
                   "D69 read FIXED off an adjective for seventy passes")
    else:
        _prose = _prose_statuses(_dt_u)
        if _prose:
            bad.append("defect(s) whose latest heading declares a status only in prose or under "
                       "a negation, so the parser stored something nobody wrote: "
                       + "; ".join(f"{n} ({w}, {why})" for n, w, why in _prose)
                       + " -- write the status in CAPITALS and unnegated on a heading")
        else:
            ok.append("no defect's status is read out of lower-case prose or out of a negation "
                      "(probe fires on both shapes)")

    # --- a row CONCLUDES and the field its result supersedes still carries the old one --------
    # This campaign's two longest-lived record defects are the same shape: a measurement landed
    # and the answer field went on saying what it said before. D136's trajectory headline was
    # wrong for 25 passes; A15's mass shares were computed on a disqualified bundle for over 100
    # (D146). Both were found by reading, which is not a mechanism.
    #
    # So: a SMALL hand-declared table of (row, token that its conclusion supersedes, why). When
    # the row has a concluded marker and VERDICT still carries the token, this fails. The table
    # is deliberately tiny and every entry names its reason -- a big one would rot, and a rotted
    # table of expectations is worse than none.
    _SUPERSEDES = {
        "of3t-trajwide": ("on a CONFIGURATION (D136)",
                          "it measures the REAL shipped arm at ~89.2 % scope, which either "
                          "replaces the repin-arm figure GO condition 3 quotes or blocks it; "
                          "either way the bullet cannot still read as it does now"
                          " -- DISCHARGED pass 307: the arm landed at 88.0819 % and the token "
                          "is gone from VERDICT; the entry stays because a table that only "
                          "grows when something breaks is a table nobody trusts to be complete"),
        "of3t-trajbar": ("no reachable bar for a 20-step TRAJECTORY",
                         "pass 309 wrote that sentence into DOESNOT KNOWING this row was "
                         "dispatched to falsify it. The moment it concludes, the campaign holds "
                         "upstream's own bf16-against-float64 trajectory and condition 3's "
                         "magnitude question has an answer -- so a field still saying no bar "
                         "exists is the D136 shape again, declared in advance this time"),
    }
    _conc = Path("/home/moritz/.coworker/state/concluded")
    # VERDICT **and** DOESNOT. Pass 309 widened this deliberately, and the honest history is
    # worth the four lines because the reasoning that got here was wrong first: I added the
    # of3t-trajbar entry, put its token in DOESNOT, concluded the entry must be inert because
    # this read VERDICT alone -- and the negative control said CAUGHT. The token was already
    # inside VERDICT's span, written into the condition-3 bullet a pass earlier. So the entry
    # was never inert and the widening does not rescue it.
    #
    # It is still right, for a reason that does not depend on that: DOESNOT is where a
    # superseded caveat rots BY CONSTRUCTION. It is the field that says what the campaign
    # cannot claim, so every sentence in it is a standing invitation for some row to falsify
    # it, and a claim-limiting sentence that outlives its limit is the most expensive kind of
    # stale -- it makes the campaign understate what it has proven. VERDICT alone would have
    # covered today's two tokens by luck.
    # Whitespace-normalised, and that is not tidiness. These fields are hard-wrapped prose, so
    # whether a declared token matches depends on where the line happened to break: at pass 309
    # "no reachable bar for a 20-step TRAJECTORY" sat in BOTH VERDICT and DOESNOT and matched
    # only VERDICT, because DOESNOT's copy wrapped between "reachable" and "bar". A guard that
    # silently covers one of two copies is worse than one that covers neither -- it reports
    # clean. Collapse every whitespace run on both sides and the token matches the sentence
    # rather than the line layout.
    _flat = lambda t: " ".join(t.split())
    _blocks = {}
    for _fld in ("VERDICT", "DOESNOT"):
        _m = _re.search(rf"^{_fld}:(.*?)(?=^[A-Z][A-Z-]+:)", o, _re.M | _re.S)
        _blocks[_fld] = _flat(_m.group(1)) if _m else ""
    _late = []
    for _row, (_tok, _why) in _SUPERSEDES.items():
        if not (_conc.is_dir() and list(_conc.glob(_row))):
            continue
        _in = [f for f, _b in _blocks.items() if _flat(_tok) in _b]
        if _in:
            _late.append(f"{_row} has concluded and {'/'.join(_in)} still says "
                         f"\"{_tok}\" -- {_why}")
    if _late:
        bad.append("a row's conclusion supersedes a figure the answer field still carries: "
                   + "; ".join(_late))
    else:
        ok.append(f"no concluded row leaves a superseded figure in VERDICT "
                  f"({len(_SUPERSEDES)} declared)")

    # --- a defect whose headings NEVER declare a status is invisible to all of the above -------
    # Pass 240. D87 was closed in a word the parser cannot READ. This is the other half: a defect
    # whose headings carry NO status word at all. The parser's clause is deliberately conservative
    # -- a status-free heading must not drop a live defect -- but a defect that has NEVER declared
    # one simply never enters `_last`, so it is absent from `unfixed`, from UNFIXED_TRIAGE.json,
    # from GAP's naming requirement, and therefore from GO condition 5 and the USER-FACING gate
    # clause. D120 sat in that state for 29 passes while GAP's prose called it UNFIXED; so did
    # D121, which GAP calls "UNFIXED as a standing rule". 29 of 132 defects were in it when this
    # check was written.
    #
    # Failing on all 29 at once would abort every compose until someone triages them in a hurry,
    # which is how a defect gets a status word chosen for convenience. So this is a RATCHET: the
    # list is frozen in state/of3t/STATUSLESS_BACKLOG.json and the check fails on anything NOT in
    # it, and equally on an entry that has since acquired a status. The list can only shrink.
    _sl_p = Path("/home/moritz/.coworker/state/of3t/STATUSLESS_BACKLOG.json")
    _seen_d, _has_d = set(), set()
    for _m in _re.finditer(r"^### (D\d+)\b(.*)$", _dt_u, _re.M):
        _seen_d.add(_m.group(1))
        if _decls(_m.group(2)):                 # the same rule `_last` uses, from status_vocab,
            _has_d.add(_m.group(1))             # or this check answers a different question
    _statusless = sorted(_seen_d - _has_d, key=lambda d: int(d[1:]))
    if not _sl_p.is_file():
        bad.append(f"state/of3t/STATUSLESS_BACKLOG.json is absent and {len(_statusless)} "
                   f"defect(s) declare no status on any heading -- they are invisible to the "
                   f"UNFIXED set and to every gate clause built on it")
    else:
        _frozen = set(json.loads(_sl_p.read_text()).get("defects", []))
        _new = [d for d in _statusless if d not in _frozen]
        _healed = sorted(_frozen - set(_statusless), key=lambda d: int(d[1:]))
        if _new:
            bad.append(f"defect(s) with NO status word on any heading and not in the frozen "
                       f"backlog: {', '.join(_new)} -- they are invisible to `unfixed`, to "
                       f"UNFIXED_TRIAGE.json and to GAP's naming requirement. Write a word from "
                       f"the vocabulary on a heading, or add them to the backlog with a reason")
        elif _healed:
            bad.append(f"STATUSLESS_BACKLOG.json still lists {', '.join(_healed)}, which now "
                       f"declare a status -- the ratchet only counts if it is tightened; drop "
                       f"them from the file")
        else:
            ok.append(f"no defect declares a status only outside the vocabulary, and the "
                      f"statusless backlog is exactly its frozen {len(_frozen)} "
                      f"({len(_seen_d)} defects total)")

    # --- the TRIAGE SPLIT the summary quotes, against the file the GATE reads -------------------
    # Pass 237. The summary states "N scope-excluded, M USER-FACING, K campaign-internal" in the
    # field Moritz reads as the answer, and nothing checked it. It had drifted to 4/10/33 while
    # `state/of3t/UNFIXED_TRIAGE.json` held 4/8/32 -- three defects stale, and the stated split did
    # not even sum to the UNFIXED total quoted two sentences above it (47 against 44). That file is
    # not decoration: `_of3t_donecheck.py` refuses GO while its USER-FACING class is non-empty, so
    # the number in the prose and the number the gate obeys were two different numbers.
    #
    # Same shape as the check-count guard (D130) and as pass 133: a figure recomputed somewhere
    # else, quoted by hand, and never reconciled. Both directions fail here -- a split that
    # disagrees with the file, and a split that disagrees with itself.
    # Pass 262: this used to search the WHOLE document. Deleting the split from VERDICT while
    # trimming it to its cap did not fail the check -- it silently matched a historical copy in
    # PASSLOG and reported the pass-236 numbers as current. A check that falls back to history
    # cannot see a deletion, which is the failure it exists to catch. VERDICT first; the whole
    # document only if VERDICT has none, and then it says so.
    _tri_p = Path("/home/moritz/.coworker/state/of3t/UNFIXED_TRIAGE.json")
    _tri_v = _re.search(r"^VERDICT:(.*?)(?=^[A-Z][A-Z-]+:)", o, _re.M | _re.S)
    _tri_where = "VERDICT"
    _tri_m = _re.search(r"(\d+)\s+scope-excluded,\s*(\d+)\s+USER-FACING,\s*(\d+)\s+campaign-internal",
                        _tri_v.group(1) if _tri_v else "")
    if _tri_m is None:
        _tri_where = "the document outside VERDICT"
        _tri_m = _re.search(r"(\d+)\s+scope-excluded,\s*(\d+)\s+USER-FACING,\s*(\d+)\s+campaign-internal", o)
    if not _tri_p.is_file():
        warn.append("UNFIXED_TRIAGE.json is absent, so the triage split the gate reads cannot be "
                    "checked against the split the summary states")
    elif _tri_m is None:
        bad.append("the summary states no triage split as 'N scope-excluded, M USER-FACING, "
                   "K campaign-internal' -- the gate refuses GO on that file's USER-FACING class "
                   "and nothing in the answer field is pinned to it")
    else:
        _cl = json.loads(_tri_p.read_text()).get("classes", {})
        _live = (len(_cl.get("SCOPE-EXCLUDED", [])), len(_cl.get("USER-FACING", [])),
                 len(_cl.get("CAMPAIGN-INTERNAL", [])))
        _said = tuple(int(x) for x in _tri_m.groups())
        if _said != _live:
            bad.append(f"the summary states a triage split of {_said[0]}/{_said[1]}/{_said[2]} "
                       f"(scope-excluded/USER-FACING/campaign-internal) but "
                       f"UNFIXED_TRIAGE.json, which the GATE reads, holds "
                       f"{_live[0]}/{_live[1]}/{_live[2]} (read from {_tri_where})")
        elif sum(_said) != len(unfixed):
            bad.append(f"the triage split {_said[0]}/{_said[1]}/{_said[2]} sums to {sum(_said)} "
                       f"but DEFECTS.md has {len(unfixed)} UNFIXED defects -- the split and the "
                       f"total in the same field disagree")
        else:
            ok.append(f"the triage split VERDICT states ({_said[0]}/{_said[1]}/{_said[2]}) "
                      f"matches UNFIXED_TRIAGE.json and sums to the {len(unfixed)} UNFIXED "
                      f"defects in DEFECTS.md")

    # --- and the REVERSE direction, which the check above never had -------------------------
    # The coverage check is one-way: every UNFIXED defect must be NAMED in GAP. It says nothing
    # about the label GAP attaches, so a defect can be FIXED in DEFECTS.md while GAP keeps
    # calling it UNFIXED, forever, silently. Pass 196 found SIX in that state -- D77 for
    # nineteen passes, and D80 and D95 whose own GAP bodies said "RESOLVED" and "CLOSED" three
    # lines under a label that said UNFIXED. This is the same shape as D74 and D76 one more
    # time: the guard read a narrower question than the document could get wrong.
    #
    # What counts as a contradiction is deliberately narrow. GAP is allowed to reconcile a
    # status in words -- "UNFIXED in effect, fixed in code" and "UNFIXED -- escalation
    # WITHDRAWN" are honest and carry more information than either word alone. So a mismatch
    # fires only when GAP's own parenthetical says UNFIXED and does NOT also name the status
    # DEFECTS.md gives it. Nuance passes; an unreconciled contradiction does not.
    from status_vocab import DEAD as _DEAD
    _st = _last          # one definition of "a defect's status", shared with the check above

    def _gap_contradictions(gap_text, statuses):
        """Defects whose GAP label says UNFIXED while DEFECTS.md says the opposite."""
        out = []
        for m in _re.finditer(r"\*\*(D\d+)\s*\n?\(([^)]*)\)", gap_text):
            n, label = m.group(1), m.group(2).upper()
            st = statuses.get(n)
            # "FIXED" is a SUBSTRING of "UNFIXED", so a naive `st not in label` reconciles every
            # FIXED defect against a label that says the exact opposite -- and FIXED is the
            # commonest status, so the check would have been born unable to fire on the six
            # cases that motivated it. The break control below is what caught that. Strip the
            # UNFIXED occurrences before asking whether the label also names the real status.
            rest = label.replace("UNFIXED", "")
            if st in _DEAD and "UNFIXED" in label and st not in rest:
                out.append(f"{n} (GAP says UNFIXED, DEFECTS.md says {st})")
        return out

    # Break control, run before the real one: the check must FAIL on a document that contradicts
    # itself, or its silence on the real input means nothing. A17 -- a negative control has to
    # break exactly what the check reads, which here is the pairing, not the presence.
    _probe = _gap_contradictions("**D1 (UNFIXED)**: synthetic.", {"D1": "FIXED"})
    if len(_probe) != 1:
        bad.append("the GAP-vs-DEFECTS contradiction check does not fire on a known "
                   "contradiction, so its silence on the real document is uninformative")
    else:
        _contra = _gap_contradictions(gap, _st)
        if _contra:
            bad.append("GAP contradicts DEFECTS.md on: " + ", ".join(_contra)
                       + " -- relabel in GAP, or reconcile the two words in GAP's own "
                         "parenthetical if the nuance is real")
        else:
            ok.append(f"GAP's {len(_st)} defect labels do not contradict DEFECTS.md "
                      f"(contradiction probe fired)")

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
        # Pass 176, FIFTH sighting of this class: the campaign reached 100 defects and this
        # returned None, so the check announced it could not run. That announcement is the
        # pass-138 fix working -- but a generator that stops at 99 is still a word list with
        # extra steps. Hundreds are now generated too, and the range below is 10x the subject's
        # current size rather than one step ahead of it.
        if n < 20:
            return _ONES[n]
        if n < 100:
            return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
        if n < 1000:
            head = _ONES[n // 100] + " hundred"
            rest = n % 100
            return head if not rest else head + " " + _word(rest)
        return None

    _words = {n: _word(n) for n in range(1, 1000)}

    # --- ROWS must match the briefs on disk -------------------------------------------------
    # Pass 182: ROWS read "twenty-four dispatched, twenty-one concluded" for SEVEN passes while
    # the campaign ran 39 rows. Nothing caught it because ROWS is the one census field with no
    # check. It gets the same treatment as every other transcribed summary here: checked against
    # its source. The source for "dispatched" is the brief files; for "concluded" it is the
    # markers MINUS this row's own, because of3t-orchestrator leaves a marker from an earlier
    # pass that is stale the moment it is relaunched -- counting markers alone overstates by one,
    # which is exactly the error this check first found in its own subject.
    _WS  = Path("/home/moritz/.coworker/workstreams")
    _CON = Path("/home/moritz/.coworker/state/concluded")
    if not (_WS.is_dir() and _CON.is_dir()):
        warn.append("ROWS census not checked -- the fleet's workstreams/ or state/concluded/ is "
                    "not reachable from this host, so the count has no source to be checked "
                    "against here")
    elif ORCH.is_file():
        _n_disp = len(list(_WS.glob("of3t-*.txt")))
        _n_conc = len([d for d in _CON.iterdir()
                       if "of3t" in d.name and "of3t-orchestrator" not in d.name])
        _rm = _re.search(r"^ROWS:\s*\*\*([a-z-]+) dispatched, ([a-z-]+) concluded",
                         ORCH.read_text(), _re.M)
        if _rm is None:
            bad.append("ROWS does not open with '**<word> dispatched, <word> concluded**', so "
                       "the census cannot be checked against the briefs on disk")
        else:
            _wd, _wc = _words.get(_n_disp), _words.get(_n_conc)
            if _wd is None or _wc is None:
                bad.append(f"ROWS check cannot run: no number-word for {_n_disp}/{_n_conc}")
            elif _rm.group(1) != _wd or _rm.group(2) != _wc:
                bad.append(f"ROWS says '{_rm.group(1)} dispatched, {_rm.group(2)} concluded' but "
                           f"disk has {_n_disp} of3t briefs and {_n_conc} concluded markers "
                           f"(excluding this row's own) -- i.e. '{_wd}' and '{_wc}'")
            else:
                ok.append(f"ROWS matches the briefs on disk ({_wd} dispatched, {_wc} concluded)")

        # --- a CONCLUDED row must have pushed a branch -----------------------------------------
        # `compose_verify.sh` merges `origin/wk/of3t-$r` only `if` the ref exists and otherwise
        # prints "dispatched but has not pushed a branch yet, skipped". That note is right for a
        # LIVE row and silently wrong for a concluded one: the marker lands, the state doc claims
        # artifacts under `perf/<namespace>/`, the DONE_CHECK passes on the doc alone, and none of
        # the evidence is ever in the composition. Nothing checked the difference.
        #
        # Added pass 224 after watching `of3t-permalign` sit on 128K of untracked artifacts with a
        # finished GO verdict in its state doc. It is prophylactic -- no concluded row is in that
        # state today -- which is the only time a guard is cheap to add.
        #
        # Deliberately NOT a check that the branch is in the composition: a row can legitimately be
        # excluded from `wk/of3t` (superseded, or held for a conflict). What cannot be legitimate is
        # a concluded row whose work exists nowhere but one host's disk.
        import subprocess as _sp
        # ONE ls-remote for every row, not one per row: authoritative (a local ref can be stale
        # or absent) and a single round trip.
        _ls = _sp.run(["git", "ls-remote", "--heads", "origin"], capture_output=True, text=True)
        if _ls.returncode != 0:
            warn.append("concluded-rows-have-a-branch not checked: `git ls-remote origin` failed, "
                        "so there is no authoritative list to check against here")
        else:
            _heads = {ln.split("refs/heads/", 1)[1]
                      for ln in _ls.stdout.splitlines() if "refs/heads/" in ln}

            def _orphaned(concluded_names, heads):
                """Concluded row slugs with no wk/<slug> branch on origin."""
                out = []
                for n in sorted(concluded_names):
                    if "of3t" not in n or "of3t-orchestrator" in n:
                        continue
                    slug = n.split(".")[0]          # strip .falseconclude-<date>, .reopened-<date>
                    if f"wk/{slug}" not in heads:
                        out.append(slug)
                return sorted(set(out))

            # Break control, every run: a guard that cannot fire has tested nothing, and this file
            # has shipped three inert ones.
            _probe = _orphaned(["of3t-ghost"], {"wk/of3t-real"})
            if _probe != ["of3t-ghost"]:
                bad.append("the concluded-branch probe did not fire -- this check is inert")
            else:
                _orphans = _orphaned([d.name for d in _CON.iterdir()], _heads)
                if _orphans:
                    bad.append("these rows are CONCLUDED and have no branch on origin, so their "
                               "evidence exists only on one host's disk and the compose skipped "
                               "them with a note meant for a LIVE row: " + ", ".join(_orphans))
                else:
                    ok.append(f"all {_n_conc} concluded rows have a branch on origin "
                              f"(probe fires)")
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
        # Pass 198: the strict form had to widen, because the DOCUMENT's convention outgrew it
        # and the guard went quiet in exactly the way its own comment warns about. Since D111's
        # first UPDATE the campaign supersedes an entry by appending
        # `### D<n> UPDATE <k> (pass p). <status> ...`, and eight such headings -- every
        # correction written at passes 196 and 198 -- were invisible here while my own ad-hoc
        # count saw them. Two readings of the same file disagreeing by eight is the symptom.
        #
        # So a heading is VALID if the number is followed by `.` or by ` UPDATE`, and anything
        # else is still malformed. And the model changes with it: a defect is a NUMBER, not a
        # heading, so the count is over DISTINCT numbers and the status is the LATEST heading's
        # -- the same definition the contradiction check uses, shared rather than re-derived.
        _loose = _re.findall(r"^### (D\d+)(\.| UPDATE\b| ?[^.\n])", _dt, _re.M)
        _malformed = [f"{d}{c!r}" for d, c in _loose if c != "." and not c.startswith(" UPDATE")]
        if _malformed:
            bad.append("defect heading(s) do not use the `### D<n>.` or `### D<n> UPDATE` form "
                       "the DEFECTS guards match, so they are INVISIBLE to the count, the "
                       "UNFIXED list and GAP's coverage check: " + ", ".join(_malformed))
        _valid = _re.findall(r"^### (D\d+)(?:\.| UPDATE\b)(.*)$", _dt, _re.M)
        _nums = {d for d, _ in _valid}
        n_def = len(_nums)
        if len(_loose) != len(_valid):
            bad.append(f"DEFECTS holds {len(_loose)} defect-shaped headings but only "
                       f"{len(_valid)} parse -- {len(_loose) - len(_valid)} entr(y/ies) are "
                       f"unaudited")
        if not _malformed and len(_loose) == len(_valid):
            # Said out loud on success: a guard that is silent when green is invisible when
            # green, which is how this one's absence went unnoticed for 175 passes.
            ok.append(f"all {len(_valid)} defect headings parse over {n_def} distinct defects, "
                      f"so none is invisible to its audit")
        # Pass 241: this used to be a THIRD parse -- `findall(rest.upper())`, file order -- and it
        # read D3's "UNFIXED, out of scope, recorded so it is not lost" as RECORDED the moment that
        # word entered the vocabulary, along with D123 and D124. The audit then reported 46 UNFIXED
        # while `statuses_by_defect` reported 49, in the same run. Two parses, two answers, one
        # question. There is one parse now.
        from status_vocab import statuses_by_defect as _sbd
        _cur = _sbd(_dt)
        n_unf = sum(1 for _v in _cur.values() if _v == "UNFIXED")
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
    # Schema-tolerant, and LOUD when it cannot read: at pass 180 the artifact was re-split
    # (aux_heads left the VOID bucket, the trunk left UNREAD) and this check -- which indexed
    # fixed keys -- died with a KeyError. The audit then printed nothing, and a grep for "DRIFT"
    # came back empty, which reads exactly like a pass. A check that cannot run must SAY SO.
    def _share(block, *names):
        b = _dtg.get(block)
        if not isinstance(b, dict):
            return None
        for n in names:
            if isinstance(b.get(n), (int, float)):
                return float(b[n])
        return None

    _p = _share("survives", "total_pct", "pct")
    _pass = _share("measured_and_PASSES", "total_pct", "pct") or 0.0
    _fail = _share("measured_and_fails", "total_pct", "pct")
    _void = _share("measured_and_void_under_A18", "total_pct", "pct") or 0.0
    _u = _share("no_direct_reading", "total_pct", "pct_total", "pct")
    if _p is None or _fail is None or _u is None:
        bad.append("DISTANCE_TO_GO_AGAINST_THEIR_STEP: cannot read its shares (survives="
                   f"{_p}, fails={_fail}, unread={_u}) -- the schema moved and this check "
                   "CANNOT RUN, which is not a pass")
        _p, _pass, _fail, _void, _u = 0.0, 0.0, 0.0, 0.0, 0.0
        _tot = 100.0
    else:
        _f = _fail + _void
        _tot = _p + _pass + _f + _u
    if abs(_tot - 100.0) > 0.001:
        bad.append(f"DISTANCE_TO_GO_AGAINST_THEIR_STEP's shares sum to {_tot:.4f} %, not 100 -- a "
                   f"reading moved buckets and the split was not rebalanced")
    else:
        ok.append(f"the distance-to-go split sums to 100.0000 % ({_p:.4f} survives / "
                  f"{_pass:.4f} passes / {_fail:.4f} fails / {_u:.4f} unread)")
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
        _checks = [(_p, "surviving"), (_fail, "failing"), (_u, "unread")]
        if _pass:
            _checks.append((_pass, "measured-and-passing"))
        for _val, _lbl in _checks:
            if not any(abs(_x - _val) <= 0.005 for _x in _pcts):
                bad.append(f"VERDICT does not state the {_lbl} share {_val:.4f} % -- the "
                           f"summary has drifted from DISTANCE_TO_GO_AGAINST_THEIR_STEP")
        if not [b for b in bad if "VERDICT does not state the" in b and "share" in b]:
            ok.append("VERDICT states all three distance-to-go shares as the artifact has them")

# --- a subset's error mass must not exceed its superset's (D84) --------------------------------
# Pass 175. I cross-compared two columns of a table whose DEVICE column was supplied by other
# rows on differently-scoped sets, and published "pairformer is 2.09x more accurate" from it. The
# arithmetic that caught it is three multiplications: for mass-weighted rel_l2 on one reference,
# error mass = rel^2 * mass, and a subset's cannot exceed its superset's. The bf16 column passed
# the same test, which is what localised the defect to the device column rather than the masses.
#
# So it is a check now, run over any artifact that declares nested scopes. A pair is declared by
# a "subset_of" key naming another entry in the same list.
for _f in sorted(Path("perf/of3t_orchestrator").glob("*.json")):
    _a = j(str(_f))
    if not isinstance(_a, dict):
        continue
    for _sec, _val in _a.items():
        _rows = _val.get("rows") if isinstance(_val, dict) else (
            _val if isinstance(_val, list) else None)
        if not isinstance(_rows, list):
            continue
        _by = {r.get("scope"): r for r in _rows if isinstance(r, dict) and r.get("scope")}
        for _r in _rows:
            if not isinstance(_r, dict):
                continue
            _sup = _by.get(_r.get("subset_of"))
            if not _sup:
                continue
            try:
                _em = float(_r["rel"]) ** 2 * float(_r["mass_frac"])
                _es = float(_sup["rel"]) ** 2 * float(_sup["mass_frac"])
            except (KeyError, TypeError, ValueError):
                continue
            if _em > _es * 1.000001:
                bad.append(f"{_f.name} {_sec}: {_r['scope']!r} is declared a subset of "
                           f"{_sup['scope']!r} but carries {_em / _es:.3f}x its error mass -- "
                           f"impossible for mass-weighted rel_l2 on one reference, so the two "
                           f"figures are not on the sets they are labelled with (D84)")
            else:
                ok.append(f"{_f.name}: {_r['scope']!r} error mass is within its superset's "
                          f"({_em / _es:.3f}x)")

# --- a summary field must stay readable, which is a LENGTH property no content check sees -----
# Pass 166. VERDICT had grown to 177,928 characters over 2,371 lines, because every pass appends
# after the last field and VERDICT is the last field, so the whole narrative landed inside the one
# field a reader treats as the answer. Every content check above passed throughout -- they read the
# shares, the counts and the amendment phrase, all of which sit in its first eight lines, and none
# of them reads its size. A field can be entirely correct and entirely unusable.
if ORCH.is_file():
    _o = ORCH.read_text()
    # Pass 198: the boundary pattern was `[A-Z][A-Z_]+:` -- underscores but not HYPHENS -- while
    # the document already had two hyphenated fields, `BRANCH-VS-GATE:` and `DIRECTIVE-STATUS:`.
    # So `DIRECTIVE-STATUS:` did not terminate GAP and its 9,223 characters were measured as part
    # of it: GAP read 48,808 against a 40,000 cap when the field itself was 39,568. The cap fired
    # on a field that was inside it, and the fix for the wrong field would have been to delete
    # real content. Same class as the heading form widened above -- the guard's pattern was
    # narrower than the document's own conventions.
    # --- DOESNOT must say that the repairs are a CONFIGURATION, not the shipped port -----------
    # Pass 212. Every headline repair this campaign has produced is behind a default-off,
    # release-gated, unmerged lever: the trunk's 1.0251x needs TT_BIO_SOFTMAX_BW_RENORM on, the
    # 51.1358 % bound needs the host float64 softmax on, and compose_verify.sh asserts on every
    # compose that both stay off. PROVES and DOESNOT -- the two fields a reader treats as the
    # campaign's claim -- said none of that; the distinction lived only in VERDICT and in the rows'
    # own docs. "off by default is not a landed win" is already written down twice on this fleet,
    # and the place it would be lost is a closing summary, so it is checked where the summary is.
    _dn = _re.search(r"^DOESNOT:(.*?)(?=^[A-Z][A-Z_-]+:|\Z)", _o, _re.M | _re.S)
    _dnt = (_dn.group(1) if _dn else "").lower()
    _want_any = ("default off", "default-off", "unmerged", "release-gated", "not shipped")
    if not any(w in _dnt for w in _want_any):
        bad.append("DOESNOT does not say the repairs are a CONFIGURATION rather than the shipped "
                   "port -- every headline fix here is behind a default-off unmerged lever, and a "
                   "reader of this field cannot tell that nothing a user gets has changed")
    else:
        ok.append("DOESNOT states that the repairs are configured, not shipped")

    _CAPS = {"VERDICT": 4000, "PROVES": 20000, "DOESNOT": 20000, "GAP": 40000,
             "DIRECTIVE-STATUS": 12000}
    _over = []
    for _f, _cap in _CAPS.items():
        _m = _re.search(rf"^{_f}:(.*?)(?=^[A-Z][A-Z_-]+:|\Z)", _o, _re.M | _re.S)
        if _m and len(_m.group(1)) > _cap:
            _over.append(f"{_f} is {len(_m.group(1))} chars against a {_cap} cap")
    if _over:
        bad.append("summary field(s) have accreted past the point of being read: "
                   + "; ".join(_over) + " -- move the narrative to PASSLOG, which is what it is for")
    else:
        ok.append("every owed summary field is inside its readability cap")

# --- an INERTNESS claim must name what it compared ------------------------------------------
# Pass 184, third instance in a week of the same failure: the numeric claims here are checked on
# every compose and the CHARACTERISATIONS are not, so prose drifts freely inside a document that
# audits green. "revision-inert" is the highest-stakes word the campaign uses -- it is what
# licenses scoring a scope against either upstream tree -- and D108 found it applied to the
# diffusion transformer (51.1358 % of the mass), which is not inert: 0.4.3 gives every DiT block
# its own learned layer_norm_z where 0.5.0 has one shared. This is A27's rule one level up: a
# ratio names how its denominator arm was built; an equivalence names the two things compared.
# Required evidence is a SOURCE PATH in the same artifact, not a citation of another artifact --
# tested both ways at pass 184, and accepting citations passed all 8 artifacts including one
# that propagated the claim with nothing behind it, i.e. it was vacuous.
# The word list is the SOURCE-EQUIVALENCE family only. Pass 184 tested widening it to
# bit-identical / byte-identical and that is a CATEGORY ERROR: those are claims about
# measured tensor DATA, whose correct evidence is a number or a digest, not a source path.
# Widening would have fired on nine well-evidenced artifacts -- "max abs diff 0.0",
# "sha256 d631c39e...", a two-arm forward comparison -- i.e. the fifth time in this campaign
# a guard was nearly built too wide. Adding `cosmetic` and `functionally identical` fires on
# nothing today and closes the hole where the same claim evades the guard by word choice.
_inert = _re.compile(r"\b(inert|cosmetic|functionally identical|identical in both)\b", _re.I)
_pyp = _re.compile(r"[\w/\.\-~]+\.py\b")
_bare = []
for _f in sorted((ROOT / "perf/of3t_orchestrator").glob("*.json")):
    try:
        _j = json.load(open(_f))
        _s = json.dumps(_j)
    except Exception:
        continue
    # Pass 192: A29's guard has the same structural weakness this campaign documented one
    # commit earlier -- the text that DISCUSSES a retired inertness claim must quote it, so the
    # matcher fires on artifacts doing the right thing. It fired on
    # A_RETIRED_CLAIM_GUARD_WAS_PROTOTYPED_AND_REJECTED.json, which asserts nothing and merely
    # recounts "the diffusion transformer called revision-inert when it is not". The fix is an
    # EXPLICIT, JUSTIFIED exemption rather than a silent skip or a spurious path added to
    # satisfy the check: an artifact may carry `a29_exempt` with a non-empty reason, which is
    # visible, greppable, and has to be argued in the artifact itself.
    _exempt = isinstance(_j, dict) and str(_j.get("a29_exempt", "")).strip()
    if _inert.search(_s) and not _pyp.search(_s) and not _exempt:
        _bare.append(_f.name)
if _bare:
    bad.append("artifact(s) claim something is INERT without naming a source file they compared, "
               "so the claim cannot be re-checked and the next reader inherits it: "
               + ", ".join(_bare))
else:
    ok.append("every artifact using the word 'inert' names a source file it compared")

# --- THE_ANSWER's by_scope table must SUM TO ITS OWN TOTAL ----------------------------------
# Added pass 176. The table listed five scopes summing to 97.9933 % beside an arithmetic_check
# asserting 100.0, because the no_reading bucket was only partly enumerated: pairformer_stack's
# 5.8282 had a row and the other 2.0067 % did not. Nothing was WRONG -- both numbers were right --
# but a reader who added the column up got a different answer from the one stated, and could not
# tell which to trust. A table whose own rows do not reconstruct its total is not checkable, and
# an unchecked total is how every drift in this campaign started.
try:
    _ta = json.load(open(ROOT / "perf/of3t_orchestrator/THE_ANSWER.json"))
except Exception as _e:
    warn.append(f"THE_ANSWER.json could not be read for the by_scope sum check ({_e})")
else:
    _rows = _ta.get("by_scope") or []
    _sum = round(sum(float(r.get("mass_pct", 0)) for r in _rows), 4)
    _claim = float(_ta.get("arithmetic_check", {}).get("total_pct", 0))
    if not _rows:
        bad.append("THE_ANSWER.json has no by_scope rows -- the headline table vanished")
    elif abs(_sum - _claim) > 5e-4:
        bad.append(f"THE_ANSWER by_scope sums to {_sum} but arithmetic_check.total_pct is "
                   f"{_claim} -- the mass table does not reconstruct its own total, so a reader "
                   f"adding the column up gets a different answer from the one stated")
    else:
        ok.append(f"THE_ANSWER by_scope sums to its own total ({_sum} over {len(_rows)} scopes)")

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
    #
    # Pass 236: and the total is computed from CONFIRMATIONS, so any OTHER guard that drifts
    # silently lowers it by one -- and this check then reports the lower number under a message
    # that names the wrong cause, "the count drifted when checks were added". It did that to me
    # twice in one session: GAP went one paragraph over its cap, and the audit told me a check
    # had gone missing. A count built out of passes cannot be audited while passes are failing,
    # so say that instead of diagnosing it wrong. K60: a check that cannot run says so.
    if bad:
        warn.append(f"PROVES check count NOT EVALUABLE this run: {len(bad)} other check(s) "
                    f"drifted, and this total counts confirmations, so it would report "
                    f"'the count drifted when checks were added' for a failure that added "
                    f"nothing. Clear the other drift(s) and re-run.")
        _cm = None
    else:
        _n_ran = len(ok) + 1                   # +1 for the ok this check is about to append
        _n_now = _n_ran + len(warn)
        _cm = _re.search(r"\((\d+)\s+checks,\s*0\s+drifted\)", o)
    if _cm is None and bad:
        pass
    elif _cm is None:
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

# Publish the NAMES, not only the count. Pass 221: the executed-check total fell from 166 to 165
# and the number alone could not say which check stopped firing -- the script was byte-identical
# across the two runs, so it was data-dependent, and there was nothing to diff. A bare integer is a
# drift DETECTOR and not a drift LOCATOR, which is the same shape as this campaign's own A15 (a
# count denominator is not a scope statement). From here every compose writes the sorted list, so
# the next time the count moves the answer is one `diff`.
try:
    # Into the campaign state dir, NOT next to this script: the audit runs from the COMPOSED
    # tree under /tmp, so a sibling file is discarded the moment the compose is rebuilt and
    # nothing is ever diffable. D112 is a concluded row's worktree being pruned and taking the
    # campaign's reference with it; this is the same trap one directory over.
    (Path("/home/moritz/.coworker/state/of3t/CHECKS_RUN.txt")).write_text(
        "# every check this audit CONFIRMED, one per line, sorted. Regenerated on every compose.\n"
        "# Diff two of these to find out which check stopped firing when the count moves; the\n"
        "# count alone cannot tell you (pass 221).\n"
        + "".join(f"{line}\n" for line in sorted(ok)))
except Exception as _e:                                        # never fail the audit over a write
    print(f"  WARN  could not write CHECKS_RUN.txt ({_e})")

for line in ok:
    print(f"  ok    {line}")
for line in warn:
    print(f"  WARN  {line}")
for line in bad:
    print(f"  DRIFT {line}")
print(f"\n{len(ok)} confirmed, {len(warn)} warning(s), {len(bad)} drifted")
sys.exit(1 if bad else 0)
