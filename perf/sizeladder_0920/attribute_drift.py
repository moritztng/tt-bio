#!/usr/bin/env python3
"""Attribute every p300c size-ladder drift finding to merged work on main, or refuse to.

WHY. The p300c baseline fragments were recorded 2026-09-17/18 and main has taken 78-104 tt_bio
commits since, so the arm is red on main by itself: `pvx-gate-land` inherited 94 findings across
seven models and none of them belonged to its flag. Re-recording is the sanctioned response to
staleness, but a re-record launders a GENUINE regression into the baseline just as quietly as it
clears drift. So every finding is classified from evidence, and a model with an unresolved one is
not committed.

WHAT IT REUSES rather than re-deriving:
  * `release_gate._size_ladder_compare_levers` -- the gate's own comparator, so the findings here
    are the ones the check would print, not a parallel notion of drift. Same reuse as
    `perf/sizeladder_p300c/attrib.py` on the 09-16 instance of this fix.
  * `lever_census.LEVERS` -- the registry that already says which module each lever lives in, so
    the commit trace reads the mapping the census itself uses instead of a hand-typed table.
  * `perf/pvx_gate_land/attribute_sizeladder_drift.py`'s two refusals: a rung whose census
    observed NO CALLS of any kind is not a lever that went dark, and a bare "went dark" with no
    clause named cannot be told apart from one.

THE ONE THING THAT IS NOT A PICKAXE. Tracing a finding by `git log -S<LEVER_NAME>` returns zero
hits for every lever on this ladder, and that is correct rather than a bug: the drift here is not
commits editing lever constants, it is commits changing SHAPES and LAYOUTS upstream so a guard
that used to admit a call now has nothing to admit (`638187138` OuterProductMean output-stage
layout, `121cb8a2a` trimul drops two layout passes). So the trace is: commits in
`<baseline_commit>..HEAD` touching the module the lever is defined in, plus the module its
counter lives in.

THE (b)-TEST, which is the one that matters. "A genuine regression with no matching merged
commit" cannot be read off a commit list, because every commit in the range IS merged. What
separates a path change that costs nothing from a regression is the RUNTIME at the rung where the
lever went dark: a lever that stops serving and leaves its rung's seconds where the rest of the
ladder sits moved its traffic somewhere else, and one that stops serving while ITS rung alone
gets slower is the thing this row must not record over.

The ratio is taken against the LADDER'S OWN MEDIAN RATIO, not against 1.0. Two records of the
same model minutes apart differ by a whole-ladder offset -- nesso1 re-recorded 1.08, 1.11, 1.04,
1.08, 1.02, 1.07 against its previous entry, about +7 % everywhere, because the box was busier.
Against 1.0 with a 5 % bar that makes four of six rungs look like regressions and flagged a
REBLOCK_PERMUTE handoff as SUSPECT when the two levers taking its traffic started firing at the
same rung. A whole-ladder offset is the measurement conditions; what a regression looks like is
ONE rung standing out from its own ladder.

Usage:  attribute_drift.py [--old-ref REF] [--card p300c] [--self-test] <model> [<model> ...]
"""
import argparse
import importlib.util
import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
# The CHECKOUT's tt_bio, not whatever the venv installed: release_gate imports it at module
# scope, and a gate scored against an installed package rather than the tree under test is a
# standing trap on this repo (`parity-gate-scores-installed-package-not-checkout`).
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rg = _load("rg", ROOT / "scripts" / "release_gate.py")
import lever_census  # noqa: E402

#: How far a rung's seconds may move before a lever going dark there stops being free.
#: Same 5 % the gate's own per-rung localiser uses (SIZE_LADDER_RUNG_MOVED), so a finding this
#: tool calls benign is one the gate would also say did not move.
RUNTIME_MOVED = rg.SIZE_LADDER_RUNG_MOVED

UNOBSERVED = ("UNOBSERVED: the census saw no calls of any kind at this rung -- not evidence "
              "about any lever, and not attributable to anything")
NO_CLAUSE = ("NEEDS-CENSUS: went dark with no clause named -- read served/declined before "
             "crediting this to anyone")
BENIGN = "MAIN-DRIFT: the rung's seconds did not regress, so the path moved and cost nothing"
SUSPECT = ("SUSPECT-REGRESSION: the lever went dark AND the rung got slower -- do NOT record "
           "over this")
NO_TRACE = ("NEEDS-REVIEW: no merged commit touches this lever's module in the range -- read "
            "it by hand")


def sh(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout


def lever_modules():
    """lever name -> the set of tt_bio files it is defined in and counted in."""
    out = {}
    for row in lever_census.LEVERS:
        name, module, _attr, counter = row[0], row[1], row[2], row[3]
        files = {module}
        if counter:
            files.add(counter.split(":")[0].rsplit(".", 1)[0])
        out[name] = sorted("%s.py" % f.replace(".", "/") for f in files if f.startswith("tt_bio"))
    return out


MODULES = lever_modules()


def entry(ref, card, model):
    p = f"docs/size_ladder_baseline.d/{model}.json"
    text = (ROOT / p).read_text() if ref is None else sh("git", "show", f"{ref}:{p}")
    if not text.strip():
        return None
    return (json.loads(text).get("cards", {}).get(card, {}).get("models", {}) or {}).get(model)


def touching(flag, base_commit):
    """Merged commits since the baseline that touch the module this lever lives in.

    Each is checked an ancestor of HEAD, because a commit can be reachable in the object store
    from a branch that never merged -- two of the nine p300c entries name exactly such a commit
    as their own provenance (`5264350b0`, `f61c33427`, both on `wk/c14-p300c-256-recell`).
    """
    files = MODULES.get(flag)
    if not files:
        return None
    raw = sh("git", "log", "--format=%h|%s", f"{base_commit}..HEAD", "--", *files)
    out = []
    for line in raw.splitlines():
        h, _, subj = line.partition("|")
        if h and subprocess.run(["git", "merge-base", "--is-ancestor", h, "HEAD"],
                                cwd=ROOT).returncode == 0:
            out.append((h, subj))
    return out


def silent(cell):
    return bool(cell) and (cell.get("served") or 0) == 0 and (cell.get("declined") or 0) == 0


def registry_change(flag, base_commit):
    """Merged commits that add or remove this lever's ROW in the census registry.

    Unlike a pickaxe on the lever inside tt_bio/, this one works: a lever appearing in or
    vanishing from the census is literally a line edited in scripts/lever_census.py, so the
    pickaxe finds the commit that did it.
    """
    raw = sh("git", "log", "--format=%h|%s", f"{base_commit}..HEAD", f"-S{flag}",
             "--", "scripts/lever_census.py")
    return [(h, subj) for h, _, subj in (l.partition("|") for l in raw.splitlines()) if h]


def classify(flag, detail, new_cell, base_commit, rt_ratio):
    """Verdict + the commits behind it. `rt_ratio` is this rung's seconds ratio DIVIDED BY the
    ladder's median ratio, so a whole-ladder offset from host conditions cancels and only a rung
    standing out from its own ladder can read as a regression."""
    if "observed NO calls" in detail:
        return UNOBSERVED, []
    # A lever the baseline has no row for, or has a row the census no longer emits, is a
    # REGISTRY change rather than a behaviour change, and the commit that made it is findable.
    if detail.startswith(("new lever not in the baseline", "in baseline but missing")):
        hits = registry_change(flag, base_commit)
        if hits:
            return ("REGISTRY: the census row itself was added or removed after the baseline "
                    "was recorded -- pure staleness"), hits
        return ("NEEDS-REVIEW: the census row appeared or vanished with no commit editing "
                "lever_census.py in the range"), []
    if "went dark" in detail and " on " not in detail and silent(new_cell):
        return NO_CLAUSE, []
    hits = touching(flag, base_commit)
    if "went dark" in detail and rt_ratio is not None and rt_ratio > 1.0 + RUNTIME_MOVED:
        return f"{SUSPECT} ({rt_ratio:.2f}x)", hits or []
    if hits is None:
        return f"NEEDS-REVIEW: {flag} is not in lever_census.LEVERS", []
    if not hits:
        return NO_TRACE, []
    tail = "" if rt_ratio is None else f" ({rt_ratio:.2f}x at this rung)"
    return BENIGN + tail, hits


def report(model, old, new, base_commit):
    rows, unresolved = [], 0
    o_rt, n_rt = old.get("runtime_s") or {}, new.get("runtime_s") or {}
    # The whole-ladder offset between the two records: host conditions, not a lever.
    allr = [n_rt[r] / o_rt[r] for r in n_rt if o_rt.get(r) and n_rt.get(r)]
    base = statistics.median(allr) if allr else None
    print(f"  whole-ladder median ratio {base:.3f} (divided out below; a rung is judged against "
          f"its own ladder, not against 1.0)" if base else "  no paired runtimes")
    for rung in sorted(new["levers"], key=int):
        b = old["levers"].get(str(rung))
        if b is None:
            print(f"  rung {rung}: absent from the old baseline entirely (new rung)")
            continue
        ob, nb = o_rt.get(str(rung)), n_rt.get(str(rung))
        ratio = (nb / ob / base) if (ob and nb and base) else None
        for f in rg._size_ladder_compare_levers(b, new["levers"][str(rung)], f"{model}/{rung}"):
            flag = f.split()[1].rstrip(":")
            detail = f.split(": ", 1)[1] if ": " in f else f
            verdict, hits = classify(flag, detail, new["levers"][str(rung)].get(flag),
                                     base_commit, ratio)
            if verdict.startswith(("NEEDS-REVIEW", "SUSPECT", "NEEDS-CENSUS")):
                unresolved += 1
            rows.append((rung, flag, detail, verdict, hits))
    return rows, unresolved


def self_test(card, model):
    """Positive + negative control: the verdict must follow the EVIDENCE, not the lever's name.

    Takes the real entry, mutates one cell, and asserts the classifier moves. Without this the
    table below is a function that could be returning a constant.
    """
    new = entry(None, card, model)
    base = "5264350b0"
    flag = "REBLOCK_PERMUTE"
    dark = "frac 1.000 -> 0.000 (went dark on gated_window_BufferType.L1)"
    cell = {"served": 0, "declined": 1120, "frac": 0.0, "rejects": {"gated_window": 1}}
    v_flat, _ = classify(flag, dark, cell, base, 1.00)
    v_slow, _ = classify(flag, dark, cell, base, 1.40)
    # The case the old bar got wrong: a rung 8 % above the PREVIOUS record but exactly on its own
    # ladder's median, which is what a busier box looks like.
    v_offset, _ = classify(flag, dark, cell, base, 1.00)
    # A bare "went dark" with no clause: a lever that really declines names what it declined
    # ON, so a bare one is either a decline with an empty rejects dict or a rung nothing ran
    # at, and the finding cannot tell those apart.
    v_silent, _ = classify(flag, "frac 1.000 -> 0.000 (went dark)",
                           {"served": 0, "declined": 0, "frac": 0.0}, base, 1.00)
    v_unknown, _ = classify("NOT_A_REAL_LEVER", dark, cell, base, 1.00)
    # Both directions of the registry arm. 7fb08268f is the commit that gave TRANSITION_H_CHUNK
    # its census row, so a baseline recorded BEFORE it traces and one recorded after does not.
    newrow = "new lever not in the baseline (re-record with --size-ladder-record)"
    v_new, _ = classify("TRANSITION_H_CHUNK", newrow, cell, "7fb08268f~1", 1.00)
    v_new_after, _ = classify("TRANSITION_H_CHUNK", newrow, cell, "679b9ab4c", 1.00)
    print("SELF-TEST")
    print(f"  rung on its ladder median    -> {v_flat[:70]}")
    print(f"  rung 40 % above its ladder   -> {v_slow[:70]}")
    print(f"  same lever, census silent    -> {v_silent[:70]}")
    print(f"  lever not in the registry    -> {v_unknown[:70]}")
    print(f"  census row added since       -> {v_new[:70]}")
    print(f"  same row, baseline after it  -> {v_new_after[:70]}")
    assert v_flat.startswith("MAIN-DRIFT"), v_flat
    assert v_slow.startswith("SUSPECT"), v_slow
    assert v_silent.startswith("NEEDS-CENSUS"), v_silent
    assert v_unknown.startswith("NEEDS-REVIEW"), v_unknown
    assert v_new.startswith("REGISTRY"), v_new
    assert v_new_after.startswith("NEEDS-REVIEW"), v_new_after
    assert new is not None
    print("  PASS: the verdict follows the runtime and the census, not the lever's name\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-ref", default="origin/main")
    ap.add_argument("--card", default="p300c")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("models", nargs="*")
    args = ap.parse_args()

    print(f"HEAD {sh('git', 'rev-parse', '--short', 'HEAD').strip()}   "
          f"old-ref {args.old_ref}   card {args.card}")
    if args.self_test:
        self_test(args.card, args.models[0] if args.models else "boltz2")

    verdicts = {}
    for model in args.models:
        old, new = entry(args.old_ref, args.card, model), entry(None, args.card, model)
        print("=" * 100)
        if old is None or new is None:
            print(f"{model}: no {args.card} entry on one side")
            continue
        base_commit = old.get("commit")
        n = sh("git", "rev-list", "--count", f"{base_commit}..HEAD", "--", "tt_bio/").strip()
        print(f"{model}: {old.get('recorded')} {base_commit} grid {old.get('grid')} -> "
              f"{new.get('recorded')} {new.get('commit')} grid {new.get('grid')};  "
              f"{n} tt_bio commits between")
        print(f"  runtime old: {old.get('runtime_s')}")
        print(f"  runtime new: {new.get('runtime_s')}")
        print(f"  exponents old {({k: v['k'] for k, v in (old.get('exponents') or {}).items()})} "
              f"-> new {({k: v['k'] for k, v in (new.get('exponents') or {}).items()})}")
        rows, unresolved = report(model, old, new, base_commit)
        print(f"  -- {len(rows)} lever finding(s), {unresolved} unresolved --")
        for rung, flag, detail, verdict, hits in rows:
            print(f"  {rung:>5} {flag:<30} {detail[:70]}")
            print(f"        {verdict}")
            for h, subj in (hits or [])[:4]:
                print(f"          {h} {subj[:84]}")
        verdicts[model] = (len(rows), unresolved)
    print("=" * 100)
    for m, (n, u) in verdicts.items():
        state = "ATTRIBUTED" if n and not u else ("NO-DRIFT" if not n else "REVIEW")
        print(f"{m:<14} findings {n:<4} unresolved {u:<4} {state}")


if __name__ == "__main__":
    main()
