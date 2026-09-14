#!/usr/bin/env python3
"""Score the ship branch's 44-leg parity gate against main's, leg by leg, on digits.

The rule the ladder uses: a leg that is not PASS on main is a gap this branch INHERITED, and it
must reproduce main's digits. A leg that is PASS on main must still be PASS. Anything else is a
regression this branch caused.

There are TWO control reports and neither one alone is main. `gate_asg.json` scored
`wk/b2z2-asg-ship` @ 2f5072d88, which was main until the morning of 2026-09-13;
`gate_trunkship.json` scored `wk/b2z2-trunk-byte-round2-ship`, the branch main took next. Current
main is the merge of both, plus `wk/b2z2-pwa-residency-ship` and the host thread-pool row, so no
report was run on exactly the tree main is now, and producing one costs a 44-leg control arm.

What makes that acceptable is that the two reports AGREE: same verdict on all 44 legs, and the six
non-PASS legs reproduce each other digit for digit across two trees that differ by four kernel-level
flags. So the bar is agreement with the control consensus, and a leg the two controls disagree on is
reported rather than silently resolved. The comparison is on the `detail` string, which carries the
digits the gate scored with; both control worktrees are gone.

Usage:  gate_compare.py --ship perf/b2z2_gate/gate_qchunkship_merged.json
        gate_compare.py --partial perf/b2z2_gate/qchunkship_merged_work   # mid-run
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CONTROLS = [Path("perf/b2z2_gate/gate_asg.json"), Path("perf/b2z2_gate/gate_trunkship.json")]


def by_leg(report: Path) -> dict:
    return {r["leg"]: r for r in json.loads(report.read_text())["legs"]}


def control() -> tuple[dict, list]:
    """The consensus of the control reports, and the legs they do not agree on.

    A leg joins the consensus when every control gives it the same verdict and -- for a non-PASS
    leg, where the digits are the thing being reproduced -- the same detail string."""
    reps = [by_leg(p) for p in CONTROLS]
    consensus, disagree = {}, []
    for leg in sorted(set().union(*(set(r) for r in reps))):
        rows = [r.get(leg) for r in reps]
        if any(r is None for r in rows):
            disagree.append((leg, "absent from a control"))
        elif len({r["verdict"] for r in rows}) > 1:
            disagree.append((leg, " vs ".join(sorted({r["verdict"] for r in rows}))))
        elif rows[0]["verdict"] != "PASS" and len({r["detail"] for r in rows}) > 1:
            disagree.append((leg, "same verdict, different digits"))
        else:
            consensus[leg] = rows[0]
    return consensus, disagree


RESUMED = re.compile(r"^\(resumed from ")


def _floats(node, out):
    """Every number anywhere in a raw leg report."""
    if isinstance(node, dict):
        for v in node.values():
            _floats(v, out)
    elif isinstance(node, list):
        for v in node:
            _floats(v, out)
    elif isinstance(node, bool):
        out.append(node)
    elif isinstance(node, (int, float)):
        out.append(float(node))
    return out


def digits_reproduced(control_detail: str, raw: dict) -> tuple[bool, str]:
    """Does the raw leg report still carry every number the control's detail line printed?

    A leg the gate RESUMED reports ``detail='(resumed from <leg>.json)'`` -- the resume marker
    replaces the digits, so comparing detail strings calls every resumed non-PASS leg a
    regression. The digits themselves are still in the workdir's raw report, at full precision,
    so scoring falls back to them: each ``key=value`` the control printed must appear in the raw
    report, matched at the precision the control printed it to. The control worktrees are gone,
    which is why the control side stays a string.
    """
    vals = _floats(raw, [])
    misses = []
    pairs = re.findall(r"([A-Za-z_][\w]*)=(-?\d+\.?\d*|True|False)", control_detail)
    if not pairs:
        return False, "control detail carries no key=value digits to check"
    for key, tok in pairs:
        if tok in ("True", "False"):
            if (tok == "True") not in [v for v in vals if isinstance(v, bool)]:
                misses.append(f"{key}={tok}")
            continue
        want = float(tok)
        dp = len(tok.split(".")[1]) if "." in tok else 0
        if not any(not isinstance(v, bool) and round(v, dp) == want for v in vals):
            misses.append(f"{key}={tok}")
    if misses:
        return False, "not found in the raw report: " + ", ".join(misses)
    return True, f"{len(pairs)}/{len(pairs)} digits reproduce at printed precision"


def compare(ship: Path, workdir: Path | None, reruns: list[Path]) -> int:
    ctl, disagree = control()
    shp = by_leg(ship)
    # A leg the full run lost to host contention is re-run alone on a quiet card. Its
    # result supersedes the timed-out row, and the row it produces says where it came
    # from -- a substitution nobody can see is a substitution nobody can check.
    relaunched = {}
    for r in reruns:
        for leg, row in by_leg(r).items():
            relaunched[leg] = r.name
            shp[leg] = row
    rows, bad = [], 0
    for leg, why in disagree:
        rows.append((leg, "CONTROL-DISAGREE", why + " -- not scored here"))
    for leg in sorted(set(ctl) | set(shp)):
        c, s = ctl.get(leg), shp.get(leg)
        if c is None or s is None:
            rows.append((leg, "MISSING", f"in control={c is not None} in ship={s is not None}"))
            bad += 1
        elif c["verdict"] != s["verdict"]:
            rows.append((leg, "VERDICT-MOVED", f"{c['verdict']} -> {s['verdict']}"))
            bad += 1
        elif c["verdict"] != "PASS" and c["detail"] != s["detail"]:
            # A resumed leg's detail is the resume marker, not its digits. Score the raw report.
            raw_path = (workdir / (s.get("report") or f"{leg}.json")) if workdir else None
            if RESUMED.match(s["detail"] or "") and raw_path and raw_path.exists():
                ok, why = digits_reproduced(c["detail"], json.loads(raw_path.read_text()))
                rows.append((leg, "OK" if ok else "DIGITS-MOVED",
                             f"{s['verdict']} (resumed; {why})"))
                bad += not ok
            else:
                rows.append((leg, "DIGITS-MOVED", f"{c['detail']!r} -> {s['detail']!r}"))
                bad += 1
        else:
            src = relaunched.get(leg)
            rows.append((leg, "OK", s["verdict"] + (f" (from {src})" if src else "")))
    for leg, verdict, note in rows:
        print(f"{verdict:14s} {leg:28s} {note}")
    print(f"\n{len(rows)} legs, {len(disagree)} the controls disagree on, "
          f"{bad} not reproducing main")
    print("GATE: REPRODUCES MAIN" if bad == 0 else "GATE: DOES NOT REPRODUCE MAIN")
    return 0 if bad == 0 else 1


def partial(workdir: Path) -> int:
    """Mid-run the raw per-leg reports exist before the summary does. Every leg's verdict can be
    read from its raw report; the non-PASS legs additionally get their raw metrics printed beside
    main's detail line, because that is where a regression would show first."""
    ctl, _ = control()
    done = sorted(p.stem for p in workdir.glob("*.json") if p.stem != "GATE_CODE")
    moved = 0
    print(f"{len(done)}/{len(ctl)} legs have a report")
    for leg in done:
        c = ctl.get(leg)
        if c is None:
            print(f"UNKNOWN-LEG    {leg}")
            moved += 1
            continue
        raw = json.loads((workdir / f"{leg}.json").read_text())
        # Only the envelope/structure legs carry their own verdict. The rest are scored by the
        # gate when it writes the summary, so mid-run they are PRESENT and nothing more; claiming
        # a verdict for them here would be inventing one.
        v = raw.get("verdict") if isinstance(raw, dict) else None
        if v is None:
            print(f"{'PRESENT':14s} {leg:28s} main={c['verdict']:26s} ship=unscored-until-summary")
            continue
        ok = v == c["verdict"]
        moved += not ok
        print(f"{'OK' if ok else 'VERDICT-MOVED':14s} {leg:28s} main={c['verdict']:26s} ship={v}")
        if c["verdict"] != "PASS":
            print(f"               main detail: {c['detail']}")
            print(f"               ship raw   : {json.dumps(raw.get('metrics', raw))[:300]}")
    print(f"\n{len(done)} scored so far, {moved} with a moved verdict")
    return 0 if moved == 0 else 1



def _walk(node, path=""):
    """Every leaf in a raw leg report, keyed by its path, so a diff can name what moved."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, node


def against(ship_dir: Path, ctl_dir: Path) -> int:
    """Score two gate workdirs leg by leg on their RAW reports, at full precision.

    The control() consensus above is a pair of reports from trees main has since moved past, so a
    non-PASS leg's digits drift with main and a real match reads as DIGITS-MOVED. When a gate has
    been run on a tree that brackets the branch, its raw reports are the better control: both
    sides are full-precision JSON, so the comparison is exact equality rather than a detail string
    matched at the precision it happened to be printed to.

    GATE_CODE.json is the tree fingerprint and is EXPECTED to differ; report.json is the summary.
    """
    skip = {"GATE_CODE", "report"}
    ship = {q.stem for q in ship_dir.glob("*.json")} - skip
    ctl = {q.stem for q in ctl_dir.glob("*.json")} - skip
    both = sorted(ship & ctl)
    bad = 0
    print(f"ship {len(ship)} legs, control {len(ctl)} legs, {len(both)} comparable\n")
    if not both:
        # A comparison of nothing is not a match. Pointing --partial at a LOG instead of the
        # gate WORKDIR globs zero reports and used to print IDENTICAL and exit 0, which is a
        # pass no evidence backs.
        print("RAW: NOTHING COMPARED -- --partial must be the gate WORKDIR holding the raw "
              "per-leg <leg>.json reports, not a log or a summary")
        return 2
    for leg in both:
        a = dict(_walk(json.loads((ship_dir / f"{leg}.json").read_text())))
        b = dict(_walk(json.loads((ctl_dir / f"{leg}.json").read_text())))
        keys = sorted(set(a) | set(b))
        diffs = [(k, b.get(k, "<absent>"), a.get(k, "<absent>"))
                 for k in keys if a.get(k) != b.get(k)]
        if not diffs:
            print(f"{'IDENTICAL':14s} {leg:28s} {len(keys)} leaves equal at full precision")
        else:
            bad += 1
            print(f"{'DIFFERS':14s} {leg:28s} {len(diffs)}/{len(keys)} leaves moved")
            for k, cv, sv in diffs[:6]:
                print(f"               {k}: control={cv} ship={sv}")
    print(f"\n{len(both)} legs compared, {bad} differing")
    print("RAW: IDENTICAL TO CONTROL" if bad == 0 else "RAW: DIFFERS FROM CONTROL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", type=Path)
    ap.add_argument("--partial", type=Path)
    ap.add_argument("--workdir", type=Path,
                    help="the gate workdir holding the raw per-leg reports. A leg the gate "
                         "RESUMED reports the resume marker as its detail, not its digits; "
                         "this is where the digits still are.")
    ap.add_argument("--rerun", type=Path, action="append", default=[],
                    help="a single-leg gate summary that supersedes the same leg in "
                         "--ship. A leg the full run lost to host contention is re-run "
                         "alone on a quiet card; this is where its result comes from.")
    ap.add_argument("--against", type=Path,
                    help="a second gate WORKDIR to score --partial against on raw digits, "
                         "instead of the control consensus. Use when a gate exists on a tree "
                         "that brackets the branch: the comparison becomes exact equality.")
    a = ap.parse_args()
    if a.against:
        raise SystemExit(against(a.partial, a.against))
    raise SystemExit(compare(a.ship, a.workdir, a.rerun) if a.ship else partial(a.partial))
