"""Break the real capture on purpose and check the reducer notices.

A check that has never failed is not a check. These controls mutate one field of a COPY of the
finished run and require (a) the targeted fold to be rejected for the named reason and (b) every
other fold to stay accepted, so each control is specific rather than merely destructive. The CIFs
are copied once and never touched; only result.json is rewritten between cases.
"""
from __future__ import annotations
import copy, json, shutil, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import reduce as R

PASSED, FAILED = [], []


def apply(pristine, tmp, size, mutate):
    run = copy.deepcopy(pristine)
    label = mutate(run)
    (tmp / str(size) / "result.json").write_text(json.dumps(run, indent=2) + "\n")
    return label


def rejected_for(result, size, label, needle):
    rows = {r["label"]: r for r in result["targets"][str(size)]["rows"]}
    row = rows[label]
    hit = any(needle in x for x in row["rejections"])
    others = [r for l, r in rows.items() if l not in (label, "cold")]
    specific = all(r["accepted"] for r in others)
    return hit, specific, row["rejections"]


def case(name, pristine, tmp, size, mutate, needle):
    label = apply(pristine, tmp, size, mutate)
    res = R.reduce(tmp)
    hit, specific, why = rejected_for(res, size, label, needle)
    if hit and specific:
        PASSED.append(f"{name}: {label} rejected for {needle!r}, every other fold still accepted")
    else:
        FAILED.append(f"{name}: hit={hit} specific={specific} rejections={why}")


def first(run, clk):
    return next(r for r in run["rows"] if r["clock_MHz"] == clk and r["label"] != "cold")


def main(runroot, size=512):
    runroot = Path(runroot)
    with tempfile.TemporaryDirectory(dir="/home/ttuser/.coworker/wt/c10-fixed-cost/perf/c10_fixed_cost/tmp") as td:
        tmp = Path(td) / "run"
        shutil.copytree(runroot, tmp)
        pristine = json.loads((runroot / str(size) / "result.json").read_text())

        # control 0: untouched, everything is accepted -- otherwise the controls below prove nothing
        (tmp / str(size) / "result.json").write_text(json.dumps(pristine, indent=2) + "\n")
        base = R.reduce(tmp)
        bad = [r["label"] for r in base["targets"][str(size)]["rows"]
               if not r["accepted"] and r["label"] != "cold"]
        (PASSED if not bad else FAILED).append(
            f"pristine copy: every warm fold accepted{'' if not bad else f', but {bad} were not'}")

        def relabel_clock(run):
            r = first(run, 800); r["clock_MHz"] = 1350; return r["label"]
        case("an 800 MHz fold relabelled 1350 is caught by its own samples",
             pristine, tmp, size, relabel_clock, "clock")

        def unsettled(run):
            r = first(run, 800); r["settle"]["settled"] = False; return r["label"]
        case("an arm that never settled is rejected", pristine, tmp, size, unsettled, "settle")

        def refused(run):
            r = first(run, 1000); r["force_response"][0] = 255; return r["label"]
        case("a refused FORCE_AICLK is rejected", pristine, tmp, size, refused, "FORCE_AICLK")

        def unauthorised(run):
            r = first(run, 1200); r["clock_MHz"] = 900; return r["label"]
        case("a clock outside the authorised arms is rejected",
             pristine, tmp, size, unauthorised, "authorised")

        def foreign(run):
            r = first(run, 1350)
            r["before"]["holders"] = [{"pid": 999999, "nodes": ["/dev/tenstorrent/1"], "cmd": "foreign"}]
            return r["label"]
        case("a foreign device holder anywhere on the box rejects the whole target",
             pristine, tmp, size, foreign, "foreign holder")

        def timer(run):
            r = first(run, 1200); r["elapsed_s"] = r["elapsed_s"] - 1.0; return r["label"]
        case("an elapsed_s that disagrees with the monotonic stamps is rejected",
             pristine, tmp, size, timer, "timer mismatch")

        def route(run):
            r = first(run, 1000); r["above_cap_sdpa_counts"] = [1, 0]; return r["label"]
        case("a fold that entered the above-cap SDPA route is rejected",
             pristine, tmp, size, route, "above-cap route")

        # a short cell must sink the target verdict even though every surviving fold is clean
        run = copy.deepcopy(pristine)
        for r in run["rows"]:
            if r["clock_MHz"] == 800:
                r["valid"] = False
        (tmp / str(size) / "result.json").write_text(json.dumps(run, indent=2) + "\n")
        res = R.reduce(tmp)
        t = res["targets"][str(size)]
        ok = t["verdict"] == "STOP" and not t["cells"]["800"]["enough"]
        (PASSED if ok else FAILED).append(
            f"losing the 800 MHz cell sinks the target verdict: verdict={t['verdict']} "
            f"800_enough={t['cells']['800']['enough']}")

        # and with only one clock left the fit refuses to invent a fixed term
        run = copy.deepcopy(pristine)
        for r in run["rows"]:
            if r["clock_MHz"] != 1350:
                r["valid"] = False
        (tmp / str(size) / "result.json").write_text(json.dumps(run, indent=2) + "\n")
        t = R.reduce(tmp)["targets"][str(size)]
        ok = "fit_all_folds" not in t and t["verdict"] == "STOP"
        (PASSED if ok else FAILED).append(
            f"one surviving clock produces no fixed term at all: fit_present={'fit_all_folds' in t} "
            f"verdict={t['verdict']}")

    print("\n".join(f"PASS  {x}" for x in PASSED))
    print("\n".join(f"FAIL  {x}" for x in FAILED))
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 512))
