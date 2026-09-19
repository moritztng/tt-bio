#!/usr/bin/env python3
"""Score a size-ladder run against its baseline from the artifacts on disk. No card.

The ladder arm folds nine models at six rungs and prints its verdict at the end, ~3 h
later. Everything it needs to score a model is on disk the moment that model's last rung
finishes, so a run that is 40 minutes from done has already decided five of its nine rows
and there is no way to read them. Four ladder reds this month were adjudicated by hand out
of the log and the fragment dir, and one of them was a co-tenant rather than a defect.

``release_gate._size_ladder_compare`` is pure by construction and its docstring says why:
"so the arm's verdicts can be tested, and reproduced from a past run's numbers, without a
card". This is the driver it never got. Point it at a ``perf/sizegate/work`` directory —
live, finished or killed — and it reports every model the run has already folded, using
the gate's own comparator and the gate's own baseline reader, so a row that reads PASS here
reads PASS there.

    scripts/ladder_preread.py perf/sizegate/work
    scripts/ladder_preread.py /path/to/work --baseline /other/tree/docs/size_ladder_baseline.json

The baseline defaults to this tree's. Pass --baseline when the run you are scoring came
from a different checkout: the cells are versioned with the code and scoring a run against
the wrong tree's cells is exactly the mistake this tool exists to stop being made by hand.

A work directory is reused between runs and the arm walks its models in order, so a live
run leaves the PREVIOUS run's artifacts sitting in place for every model it has not
reached yet. Scoring those as if they were this run's is the one way this tool can lie, so
every row carries the wall-clock span of the artifacts it read and --since drops anything
older than the run you mean.

WHAT A PASS HERE DOES NOT COVER, because a pre-read that is mistaken for the verdict is
worse than no pre-read. This reproduces the COMPARISON — every lever, exemption, decline
clause, moved ceiling and exponent `_size_ladder_compare` scores. It does not reproduce the
two ways the arm can red before the comparison runs:

  * `_size_ladder_precondition(model)`, which the arm checks first and which depends on the
    folding host's environment, not on any artifact. Deliberately not called: this tool is
    meant to score a remote run from a different box, where the local answer would be the
    wrong one.
  * a fold that errored or timed out, which leaves the arm with `{"error": ..., "partial":
    True}` for that model. Here the rung simply has no artifact and reads as `pending`.

So a PASS means "nothing in this model's measurement drifted from its baseline", which is
what the arm spends three hours deciding. It does not mean the arm will print PASS if the
model never finished folding.
"""
import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_gate as rg  # noqa: E402


def _census_levers(census: dict) -> dict:
    """The lever dict `_run_census_fold` builds from a census artifact.

    Kept in step with that function by test_ladder_preread_levers_match_the_gate, which
    feeds both the same artifact — the two readers cannot silently diverge without the
    verdicts diverging with them, which is the whole value of this driver.
    """
    levers = {}
    for r in census["rows"]:
        served, declined = r["served"], r["declined"]
        total = (served or 0) + (declined or 0)
        frac = (served / total) if total else (0.0 if served == 0 else None)
        levers[r["flag"]] = {"resolved": r["resolved"], "served": served,
                             "declined": declined, "frac": frac, "how": r["how"]}
        if r.get("rejects"):
            levers[r["flag"]]["rejects"] = r["rejects"]
    return levers


def _rung_runtime(workdir: Path, model: str, rung: int, tag: str):
    """The rung's own runtime_s, read the way the arm reads it, or None.

    Globbed rather than resolved through ``tt_bio.main.predict_results_dir_name`` so this
    stays importable on a host with no tt_bio environment — the point of the tool is that
    you can score a remote run's artifacts locally.
    """
    out = workdir / f"out_{model}-{rung}-{tag}"
    if model == "nesso1":
        return rg._affinity_seconds(out)
    for results in sorted(out.glob("*/results.json")):
        try:
            rows = json.loads(results.read_text())
        except Exception:
            continue
        ts = [r["runtime_s"] for r in rows
              if r.get("status") == "ok" and r.get("runtime_s") is not None]
        if ts:
            return max(ts)
    return None


def _measure_from_disk(workdir: Path, model: str, rungs, since: float = 0.0) -> dict:
    """Rebuild `_size_ladder_measure_model`'s return value from what the run left behind.

    The arm discards the first fold at each rung, so the warm-up artifacts are read only
    for the refusal they may carry. A rung with neither a census nor a refusal has not run
    yet: it is reported as pending rather than as a hole in the measurement, because a
    partial run scored as if it were complete is how a live ladder reads as a red one.
    """
    levers, runtimes, refused, pending, sigmas = {}, {}, {}, [], {}
    grid, seen = None, []
    for rung in rungs:
        reps, i = [], 0
        while True:
            c = workdir / f"census_{model}-{rung}-rep{i}.json"
            if not c.exists():
                break
            if c.stat().st_mtime >= since:
                reps.append(f"rep{i}")
                seen.append(c.stat().st_mtime)
            i += 1
        if not reps:
            warm = workdir / f"{model}-{rung}-warmup.log"
            text = (rg._fold_log_text(warm)
                    if warm.exists() and warm.stat().st_mtime >= since else "")
            refusal = rg._size_limit_refusal(text) if text else None
            (refused.setdefault(str(rung), refusal) if refusal else pending.append(rung))
            continue
        runs = []
        for tag in reps:
            census = json.loads((workdir / f"census_{model}-{rung}-{tag}.json").read_text())
            runs.append({"levers": _census_levers(census), "grid": census.get("grid"),
                         "runtime_s": _rung_runtime(workdir, model, rung, tag)})
        ts = [r["runtime_s"] for r in runs if r["runtime_s"] is not None]
        if not ts:
            pending.append(rung)
            continue
        grid = grid or runs[0]["grid"]
        levers[str(rung)] = runs[0]["levers"]
        runtimes[str(rung)] = round(statistics.median(ts), 2)
        if len(ts) > 1:
            sigmas[str(rung)] = round(statistics.stdev(ts) / statistics.mean(ts), 4)
    return {"levers": levers, "runtime_s": runtimes, "sigma": None, "sigmas": sigmas,
            "grid": grid, "drift": [], "refused": refused, "pending": pending,
            "span": (min(seen), max(seen)) if seen else None}


def preread(workdir: Path, baseline_path: Path, models=None, card=None,
            since: float = 0.0) -> list:
    """One result row per model the run has folded at least one rung of."""
    base = rg._size_ladder_read_baseline(baseline_path)
    card = card or rg._size_ladder_card_type()
    card_block = (base.get("cards") or {}).get(card) or {}
    out = []
    for model in (models or rg.SIZE_LADDER_MODELS):
        rungs = rg._size_ladder_model_rungs(model)
        meas = _measure_from_disk(workdir, model, rungs, since)
        if not meas["levers"] and not meas["refused"]:
            continue                                  # this model has not started
        base_model = (card_block.get("models") or {}).get(model)
        if base_model is None:
            # NOT a red. "I could not find your cells" and "your measurement drifted" are
            # different answers, and printing the first as FAIL is the pre-read being
            # mistaken for the verdict, which is the one thing this tool must not do. The
            # usual cause is a --baseline pointed at a copied monolith whose
            # `<stem>.d/` fragment dir did not come with it: the fragments hold every
            # model's rows, so the merge silently yields a card block with no models and
            # every row reads as a regression.
            frag = rg._size_ladder_fragment_dir(baseline_path)
            why = (f"{model}: no {card} cells in {baseline_path.name}"
                   + ("" if frag.is_dir() else f" and no fragment dir {frag.name}/ beside it"))
            out.append({"model": model, "gate": None, "unscorable": True,
                        "pending": meas["pending"], "findings": [why],
                        "span": meas["span"], "runtime_s": meas["runtime_s"],
                        "exponents": {}})
            continue
        res = rg._size_ladder_compare(base_model, meas, model, rungs)
        res["pending"], res["span"] = meas["pending"], meas["span"]
        out.append(res)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path, help="the run's perf/sizegate/work directory")
    ap.add_argument("--baseline", type=Path, default=rg.SIZE_LADDER_BASELINE)
    ap.add_argument("--card", default=None, help="board type; probed from this host if unset")
    ap.add_argument("--models", default=None, help="comma-separated subset")
    ap.add_argument("--since", default=None,
                    help="ignore artifacts older than this (epoch or ISO8601 UTC) -- a work "
                         "dir holds the previous run's rungs for every model this one has "
                         "not reached yet")
    a = ap.parse_args()
    models = a.models.split(",") if a.models else None
    since = 0.0
    if a.since:
        try:
            since = float(a.since)
        except ValueError:
            since = datetime.fromisoformat(a.since.replace("Z", "+00:00")).timestamp()
    rows = preread(a.workdir, a.baseline, models, a.card, since)
    if not rows:
        print("no model in this workdir has a scoreable rung yet")
        return 0
    print(f"{'model':<12} {'verdict':<8} {'artifacts (UTC)':<19} {'rungs scored':<34} exponents")
    for r in rows:
        rt = " ".join(f"{k}:{v}" for k, v in sorted(r["runtime_s"].items(), key=lambda x: int(x[0])))
        exp = " ".join(f"{k}={v}" for k, v in r["exponents"].items())
        pend = f"  (pending {','.join(map(str, r['pending']))})" if r.get("pending") else ""
        span = ("%s-%s" % tuple(datetime.utcfromtimestamp(t).strftime("%H:%M")
                                for t in r["span"])) if r.get("span") else "-"
        day = (datetime.utcfromtimestamp(r["span"][0]).strftime("%m-%d ")
               if r.get("span") else "")
        verdict = "NO-CELLS" if r.get("unscorable") else ("PASS" if r["gate"] else "FAIL")
        print(f"{r['model']:<12} {verdict:<8} {day + span:<19} "
              f"{rt:<34} {exp}{pend}")
        for f in r["findings"]:
            print(f"    - {f}")
    bad = [r["model"] for r in rows if not r.get("unscorable") and not r["gate"]]
    nocell = [r["model"] for r in rows if r.get("unscorable")]
    scored = len(rows) - len(nocell)
    print(f"\n{scored - len(bad)}/{scored} scored models PASS"
          + (f"; FAIL: {', '.join(bad)}" if bad else "")
          + (f"; NOT SCORED (no cells): {', '.join(nocell)}" if nocell else ""))
    # A run nobody could score is not a green one: exit 2 so a caller can tell "your
    # baseline is missing" from "your measurement drifted" without parsing the text.
    if bad:
        return 1
    return 2 if nocell and not scored else 0


if __name__ == "__main__":
    sys.exit(main())
