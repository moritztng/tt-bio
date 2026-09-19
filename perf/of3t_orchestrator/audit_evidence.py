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

print("AUDIT of state/of3t/EVIDENCE.md against committed artifacts\n")
for line in ok:
    print(f"  ok    {line}")
for line in warn:
    print(f"  WARN  {line}")
for line in bad:
    print(f"  DRIFT {line}")
print(f"\n{len(ok)} confirmed, {len(warn)} warning(s), {len(bad)} drifted")
sys.exit(1 if bad else 0)
