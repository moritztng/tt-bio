#!/usr/bin/env python3
"""Evaluate the campaign's four charter conditions against the ARTIFACTS, not against its prose.

CPU only. No card, no network. Reads the live gate source and the committed artifacts; writes
CHARTER_EVIDENCE.json here and into `state/of3t/` for the gate to read.

WHY THIS FILE EXISTS (D122)
---------------------------
`workstreams/_of3t_donecheck.py::_charter_gate` decides whether the campaign may end. Four of its
five conditions are regexes over four fields of one markdown file that the orchestrator writes
about itself. Pass 220 showed the fifth comes apart from what it claims to test and put a ledger
clause beside it; the other four were left, and their exposure is worse, because no orchestrator
document has ever carried a THEIR-TEST, GRADIENTS, TRAJECTORY or COVERAGE field. Those clauses
have never executed. They first run on the one pass that ends the campaign.

`charter_is_a_keyword_test.py` beside this file demonstrates that rather than arguing it.

THE ONE DEFINITION IS THE GATE'S
--------------------------------
The gate must be self-contained: it runs from ~/.coworker with no tt-bio checkout, so it cannot
import from this tree. Rather than write the conditions twice -- which is the shape of half the
defects this campaign has filed against its own instruments (status_vocab.py) -- this file LIFTS
the gate's literal with `ast.literal_eval` and evaluates that. There is one definition and a
second reader that cannot drift from it, because drifting would mean failing to parse.

The published JSON carries the sha256 of the literal it was evaluated from. The gate recomputes
that sha from its own copy and refuses a file evaluated against a different one, so a condition
edited after publication invalidates the publication instead of silently outliving it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATE = Path("/home/moritz/.coworker/workstreams/_of3t_donecheck.py")
STATE = Path("/home/moritz/.coworker/state/of3t/CHARTER_EVIDENCE.json")
HERE = Path(__file__).with_name("CHARTER_EVIDENCE.json")

BEGIN, END = "# CHARTER_EVIDENCE_BEGIN", "# CHARTER_EVIDENCE_END"


def lift_spec(src: str) -> list:
    """The gate's CHARTER_EVIDENCE literal, parsed. Refuses rather than guessing."""
    if BEGIN not in src or END not in src:
        sys.exit(f"REFUSING: no {BEGIN} block in the gate source. Either the gate was changed "
                 f"(good -- re-read it) or this script has gone stale.")
    block = src.split(BEGIN, 1)[1].split(END, 1)[0]
    return ast.literal_eval(block.split("CHARTER_EVIDENCE = ", 1)[1])


def dig(obj, path: str):
    """Walk a dotted path. Returns (found, value). `*` is handled by the caller."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def test_one(doc, key, op, bar):
    """(passed, rendered_actual). `doc` is None when the artifact is absent."""
    if doc is None:
        return False, "artifact absent"
    if op == "all":
        # "a.*.covered" -- every entry of the dict at `a` must have `covered` true
        head, leaf = key.split(".*.")
        found, d = dig(doc, head)
        if not found or not isinstance(d, dict):
            return False, f"{head} is not a dict of entries"
        bad = sorted(k for k, v in d.items()
                     if not (isinstance(v, dict) and v.get(leaf) is bar))
        return (not bad), (f"{len(d) - len(bad)} of {len(d)}"
                           + (f", missing: {', '.join(bad)}" if bad else ""))
    if op == "moves":
        # "the two arms actually MOVED" -- D169. The clause this replaces read `d1`, and step 1
        # of an OpenFold3 training run cannot move: upstream's own AlphaFoldLRScheduler warms up
        # linearly from base_lr=0.0, so lr(0) == 0.0 exactly and both sides are bit-identical
        # after it even with a non-zero gradient. Read the whole trajectory instead and require
        # OUR side to move at every step where THEIRS does. A step where neither moves is
        # agreement under their schedule, not two stationary weight vectors.
        ref_k, our_k = bar
        found, rows = dig(doc, key)
        if not found or not isinstance(rows, list) or not rows:
            return False, f"{key} is not a non-empty list of steps"

        def num(r, k):
            v = r.get(k) if isinstance(r, dict) else None
            return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        ref_moving = [r for r in rows if (num(r, ref_k) or 0) > 0]
        if not ref_moving:
            return False, (f"the reference never moves in {len(rows)} step(s) -- two stationary "
                           f"weight vectors prove nothing about the update rule")
        stuck = [r.get("k") for r in ref_moving if not (num(r, our_k) or 0) > 0]
        return (not stuck), (f"{len(ref_moving)} of {len(rows)} step(s) move upstream-side"
                             + (f", ours stationary at k = {stuck}" if stuck
                                else ", ours moves at every one"))
    if op == "<=key":
        # The bar is ANOTHER PATH IN THE SAME ARTIFACT, not a literal. PROTOCOL's own §3d rule:
        # "when a reference's own replay of a scope exceeds the bar at that scope, a §3d
        # comparison against the published artifact there is void, not pessimistic -- measure the
        # floor first". A clause whose bar is a measured property of the REFERENCE cannot be
        # loosened by whoever writes the clause, which is the point.
        found, v = dig(doc, key)
        if not found:
            return False, "key absent"
        found_b, b = dig(doc, bar)
        if not found_b:
            return False, f"the bar's own key {bar} is absent from the artifact"
        for name, val in ((key, v), (bar, b)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                return False, f"{name} is {val!r}, not numeric"
        return v <= b, f"{v:.6g} against {bar} = {b:.6g}"
    found, v = dig(doc, key)
    if not found:
        return False, "key absent"
    if op == "present":
        return v is not None, ("present" if v is not None else "null")
    if op == "is":
        return v == bar and type(v) is type(bar), repr(v)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False, f"{v!r} is not numeric"
    return ((v >= bar) if op == ">=" else (v <= bar)), f"{v:.6g}"


def code_staleness(root: Path, artifact: str, code_paths=("tt_bio",)) -> dict:
    """Is this artifact OLDER than the engine code its clause is about?

    D178: `traj_shipped.json` was written 2026-09-21 02:35:02 and the `_PARAMS` re-key that fixes
    the very defect it reports landed at 02:55:54, twenty minutes later (965c24f52). The charter
    then read TRAJECTORY as NOT MET for ~22 hours on an artifact that predated its own repair, and
    nobody noticed because an artifact carries no notion of the tree it ran against. D177 was the
    same class one condition over: GRADIENTS graded the pre-D56 arm after D56 became the shipped
    default. Twice is a pattern, so it gets a reader rather than another hand-check.

    Reported, never failed. Re-taking an artifact needs a card, so a hard failure here would block
    every compose on work that cannot be done in a compose.
    """
    def ts(*args) -> int | None:
        try:
            r = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%ct", *args],
                               capture_output=True, text=True, timeout=30)
            out = r.stdout.strip()
            return int(out) if out.isdigit() else None
        except Exception:
            return None
    a = ts("--", artifact)
    c = ts("--", *code_paths)
    if a is None and (root / artifact).is_file():
        # Generated at compose time from sources pinned by digest (D179's COVERAGE_MERGED.json is
        # the first). It exists but has no commit, and that is the point: it cannot be older than
        # its inputs because it is rebuilt from them every run.
        return {"comparable": False, "generated_not_committed": True,
                "why": "composed at compose time from digest-pinned sources, so it cannot be stale"}
    if a is None or c is None:
        return {"comparable": False,
                "why": "the artifact or the code path has no commit in this tree"}
    return {"comparable": True, "artifact_commit_unixtime": a, "code_commit_unixtime": c,
            "artifact_is_older_than_code": a < c,
            "seconds_behind": max(0, c - a),
            "note": ("this artifact predates the newest commit under "
                     + "/".join(code_paths)
                     + ", so its numbers describe an earlier tree. A clause reading it is "
                       "reporting history, and a re-take may change the verdict"
                     if a < c else "the artifact is at or after the newest code commit")}


def evaluate(spec, root: Path) -> list:
    out = []
    for cond in spec:
        art = root / cond["artifact"]
        doc = None
        if art.is_file():
            try:
                doc = json.loads(art.read_text())
            except Exception as e:
                out.append({"field": cond["field"], "artifact": cond["artifact"], "met": False,
                            "misses": [f"the artifact does not parse ({e})"], "checks": []})
                continue
        checks, misses = [], []
        for key, op, bar, why in cond["require"]:
            passed, actual = test_one(doc, key, op, bar)
            checks.append({"key": key, "op": op, "bar": bar, "actual": actual,
                           "passed": passed, "why": why})
            if not passed:
                misses.append(f"{key} {op} {bar!r} -- artifact has {actual} ({why})")
        out.append({
            "field": cond["field"], "artifact": cond["artifact"], "why": cond["why"],
            "artifact_exists": art.is_file(),
            "artifact_sha256": (hashlib.sha256(art.read_bytes()).hexdigest()
                                if art.is_file() else None),
            "met": bool(checks) and all(c["passed"] for c in checks),
            "misses": misses, "checks": checks,
            "code_staleness": code_staleness(root, cond["artifact"]),
        })
    return out


def break_control(spec) -> list[str]:
    """Each clause must be able to say MET, or its UNMET is uninformative.

    A17, and the lesson D115's own first version taught: a check that cannot fire in one
    direction reports that direction forever. All four conditions are UNMET on the live tree
    and will stay UNMET for many passes, so "all four UNMET" is exactly the output a check
    that is broken in every clause would also produce. This builds, per condition, a synthetic
    artifact satisfying every requirement, and demands the evaluator flip it to MET.
    """
    bad = []
    for cond in spec:
        doc: dict = {}
        for key, op, bar, _ in cond["require"]:
            if op == "all":
                head, leaf = key.split(".*.")
                doc.setdefault(head, {})["synthetic"] = {leaf: bar}
                continue
            if op == "moves":
                ref_k, our_k = bar
                cur, parts = doc, key.split(".")
                for part in parts[:-1]:
                    cur = cur.setdefault(part, {})
                # k = 1 is the warmup no-op on BOTH sides on purpose: the synthetic artifact
                # that has to report MET is the shape a FAITHFUL reproduction produces, which is
                # the whole content of D169.
                cur[parts[-1]] = [{"k": 1, ref_k: 0.0, our_k: 0.0},
                                  {"k": 2, ref_k: 1.0, our_k: 1.0}]
                continue
            if op == "<=key":
                # both sides have to exist, and EQUAL satisfies <=
                for path, val in ((key, 1.0), (bar, 1.0)):
                    cur, parts = doc, path.split(".")
                    for part in parts[:-1]:
                        cur = cur.setdefault(part, {})
                    cur[parts[-1]] = val
                continue
            cur, parts = doc, key.split(".")
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = {"present": "a-digest", "is": bar,
                              ">=": bar, "<=": bar}[op]
        checks = [test_one(doc, k, o, b)[0] for k, o, b, _ in cond["require"]]
        if not all(checks):
            names = [k for (k, _, _, _), c in zip(cond["require"], checks) if not c]
            bad.append(f"{cond['field']}: a synthetic artifact built to satisfy every "
                       f"requirement still fails {', '.join(names)} -- this condition cannot "
                       f"report MET, so its UNMET on the real artifact means nothing")
    return bad


def negative_control(spec) -> list[str]:
    """And each clause must be able to say NOT MET for the RIGHT reason.

    `break_control` above proves a clause can report MET. That is only half of it: a clause
    hard-wired to pass would also satisfy it on a real artifact, and a clause that reports UNMET
    for a reason unrelated to its subject is worse than one that never fires. So for every
    requirement, build the artifact that violates exactly that requirement and nothing else, and
    demand the evaluator refuse it.

    Added at pass 319 with the `moves` op (D169), because that op is the first one whose failing
    case is not "a value is wrong" but "one side of the comparison stood still", and a control
    that only ever builds a passing artifact cannot tell those apart.
    """
    bad = []
    for cond in spec:
        for key, op, bar, _ in cond["require"]:
            doc: dict = {}
            if op == "all":
                head, leaf = key.split(".*.")
                doc[head] = {"synthetic": {leaf: (not bar) if isinstance(bar, bool) else None}}
            elif op == "moves":
                ref_k, our_k = bar
                cur, parts = doc, key.split(".")
                for part in parts[:-1]:
                    cur = cur.setdefault(part, {})
                # theirs moves, ours does not -- the one thing this clause exists to catch
                cur[parts[-1]] = [{"k": 1, ref_k: 0.0, our_k: 0.0},
                                  {"k": 2, ref_k: 1.0, our_k: 0.0}]
            elif op == "<=key":
                # ours WORSE than the reference's own measured floor, which is the whole subject
                for path, val in ((key, 2.0), (bar, 1.0)):
                    cur, parts = doc, path.split(".")
                    for part in parts[:-1]:
                        cur = cur.setdefault(part, {})
                    cur[parts[-1]] = val
            else:
                if op == "present":
                    v = None
                elif isinstance(bar, bool):
                    v = not bar
                elif isinstance(bar, (int, float)):
                    v = bar + 1 if op in ("is", "<=") else bar - 1
                else:
                    v = "not-the-bar"
                cur, parts = doc, key.split(".")
                for part in parts[:-1]:
                    cur = cur.setdefault(part, {})
                cur[parts[-1]] = v
            passed, actual = test_one(doc, key, op, bar)
            if passed:
                bad.append(f"{cond['field']}: an artifact built to VIOLATE `{key} {op} {bar!r}` "
                           f"still passes it (evaluator read {actual}) -- this clause cannot "
                           f"report NOT MET, so its MET would mean nothing")
    return bad


def main() -> int:
    src = GATE.read_text()
    spec = lift_spec(src)
    spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    broken = break_control(spec) + negative_control(spec)
    conds = evaluate(spec, ROOT)
    payload = {
        "instrument": "perf/of3t_orchestrator/charter/charter_evidence.py",
        "spec_lifted_from": str(GATE),
        "spec_sha256": spec_sha,
        "tree": str(ROOT),
        "break_control": ("every condition can report MET on a synthetic artifact, and every "
                          "requirement reports NOT MET on one built to violate it"
                          if not broken else broken),
        "n_met": sum(c["met"] for c in conds),
        "n_conditions": len(conds),
        "conditions": conds,
    }
    if broken:
        # Publishing an evaluation whose own controls failed would let the gate read UNMET off a
        # broken instrument. Refuse instead: the gate then reports the file missing, which is
        # true and actionable, rather than four conditions it cannot trust.
        for b in broken:
            print("BREAK CONTROL FAILED:", b)
        print("\nREFUSING to publish. Fix the evaluator before the gate reads it.")
        return 2

    for f in (HERE, STATE):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"charter evidence, spec {spec_sha[:12]} lifted from the live gate\n")
    for c in conds:
        print(f"{'MET    ' if c['met'] else 'NOT MET'}  {c['field']:11s} {c['artifact']}")
        for m in c["misses"]:
            print(f"             - {m}")
    print(f"\n{payload['n_met']} of {payload['n_conditions']} charter conditions met. "
          f"Break control: all {len(conds)} clauses can report MET on a synthetic artifact, "
          f"and every requirement reports NOT MET on one built to violate it.")
    print(f"written: {HERE}\n         {STATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
