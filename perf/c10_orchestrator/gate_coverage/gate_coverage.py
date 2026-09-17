#!/usr/bin/env python3
"""The release gate never folds Boltz-2 at production length above 117 residues.

`c10-size-scaling` wedged a Blackhole p300c on a 768 aa Boltz-2 fold at shipped settings -- 200
sampling steps, 3 recycles, a 35-row A3M -- nondeterministically, after five identical folds had
completed. 768 aa is a size a user can ask for. So: would the release gate have caught it?

No, and not because any arm is broken. Each arm does what it says. This maps the gate's Boltz-2
coverage on the two axes that matter for this failure, target size and fold length, and finds the
hole is in the composition. Everything is parsed out of `scripts/release_gate.py` and the fixtures
themselves rather than restated, so the map cannot quietly go stale against the gate it describes.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GATE = REPO / "scripts" / "release_gate.py"

# What c10-size-scaling actually ran when the chip wedged.
INCIDENT = {"size_aa": 768, "steps": 200, "recycles": 3, "msa_rows": 35,
            "folds_before_failure": 5, "card": "p300c", "row": "c10-size-scaling"}


def _const(src, name, cast=int):
    m = re.search(rf"^{name}\s*=\s*([0-9]+)", src, re.M)
    return cast(m.group(1)) if m else None


def _residues(rel):
    p = REPO / rel
    if not p.is_file():
        return None
    return sum(len(s) for s in re.findall(r"sequence:\s*([A-Za-z]+)", p.read_text()))


def _rungs(src):
    m = re.search(r"^SIZE_LADDER_RUNGS\s*=\s*tuple\(int\(x\) for x in\s*\n?\s*(.+?)\)\)", src,
                  re.M | re.S)
    nums = re.findall(r"\b(\d{3,4})\b", m.group(1)) if m else []
    return sorted({int(n) for n in nums if 100 <= int(n) <= 4096})


def analyse():
    src = GATE.read_text()
    prod_steps = _const(src, "SAMPLING_STEPS")
    ladder_steps = _const(src, "SIZE_LADDER_STEPS")
    cap_steps = _const(src, "CAPACITY_STEPS")
    l1_steps = _const(src, "L1_BUDGET_STEPS")

    main_fixture = "examples/prot.yaml"
    l1_fixture = "examples/affinity_fkg.yaml"
    legs = [
        {"leg": "accuracy (per-model fold)", "model": "boltz2", "fixture": main_fixture,
         "size_aa": _residues(main_fixture), "steps": prod_steps, "single_sequence": False},
        {"leg": "l1-budget", "model": "boltz2", "fixture": l1_fixture,
         "size_aa": _residues(l1_fixture), "steps": l1_steps, "single_sequence": True},
        {"leg": "size-ladder", "model": "boltz2", "fixture": "perf/size512/fixtures/cdk2x2_<N>",
         "size_aa": _rungs(src), "steps": ladder_steps, "single_sequence": True},
        {"leg": "capacity", "model": "protenix-v2 / opendde-abag", "fixture": "abag yamls",
         "size_aa": [1095, 891], "steps": cap_steps, "single_sequence": False},
    ]

    boltz = [l for l in legs if l["model"] == "boltz2"]
    prod = [l for l in boltz if l["steps"] == prod_steps]
    prod_sizes = [s for l in prod for s in ([l["size_aa"]] if isinstance(l["size_aa"], int)
                                            else l["size_aa"]) if s]
    above = [l for l in boltz if (max(l["size_aa"]) if isinstance(l["size_aa"], list)
                                  else l["size_aa"] or 0) > max(prod_sizes or [0])]

    out = {
        "scope": "CPU read of scripts/release_gate.py and its fixtures. No device, no gate run, "
                 "no gate change proposed as code.",
        "gate": str(GATE),
        "step_constants": {"SAMPLING_STEPS": prod_steps, "SIZE_LADDER_STEPS": ladder_steps,
                           "CAPACITY_STEPS": cap_steps, "L1_BUDGET_STEPS": l1_steps},
        "boltz2_legs": boltz,
        "all_legs": legs,
        "incident": INCIDENT,
    }

    max_prod = max(prod_sizes) if prod_sizes else 0
    out["gap"] = {
        "largest_boltz2_target_folded_at_production_steps_aa": max_prod,
        "production_steps": prod_steps,
        "legs_above_that_size": [l["leg"] for l in above],
        "their_steps": sorted({l["steps"] for l in above}),
        "step_ratio": prod_steps / ladder_steps if ladder_steps else None,
        "statement": "Every Boltz-2 target the gate folds above %d aa is folded at %d steps, "
                     "%.1fx shorter than the %d the product runs, and single-sequence so the MSA "
                     "path is not exercised there at all."
                     % (max_prod, ladder_steps, prod_steps / ladder_steps, prod_steps),
    }
    out["incident_in_the_gap"] = {
        "covered": INCIDENT["size_aa"] <= max_prod,
        "reading": "The wedge happened at %d aa and %d steps. The gate's only Boltz-2 coverage at "
                   "that size runs %d steps, roughly %.0f %% of the diffusion device work, and "
                   "single-sequence. A hang that needs production-length work to appear is "
                   "invisible to it, and this one is nondeterministic on top: five identical folds "
                   "completed first."
                   % (INCIDENT["size_aa"], INCIDENT["steps"], ladder_steps,
                      100.0 * ladder_steps / INCIDENT["steps"]),
    }
    out["not_a_criticism_of_any_arm"] = (
        "The size-ladder arm's own docstring says it counts guard decisions rather than trajectory "
        "statistics and picks six steps deliberately, and it names its own price: a cliff living "
        "only in the MSA module is invisible there. It is doing exactly what it claims. The hole is "
        "in the COMPOSITION -- size coverage and length coverage are supplied by different legs and "
        "they do not overlap, so no leg folds a large target for a long time."
    )
    out["minimal_fix_not_implemented_here"] = [
        "One long rung: fold the top size-ladder rung at production steps with an MSA, as a "
        "stability check with a timeout, scoring completion rather than counters. Cost is one fold.",
        "Or state the limit in the gate's own summary, so 'size-ladder PASS' is not read as "
        "'large targets are safe'.",
        "Either is a release-gate change and therefore gated: it stays on this branch and is "
        "Moritz's call, not this row's.",
    ]
    out["limits"] = [
        "Coverage is read from constants and fixture files. A leg that folds a large target through "
        "a path this parse does not recognise would be missed; the legs list is explicit so it can "
        "be checked by eye against the file.",
        "This says nothing about whether the wedge reproduces, what causes it, or how often. One "
        "occurrence in six folds is not a rate.",
        "rf3 folds 997 aa in the gate, but that is a different model and does not cover Boltz-2.",
    ]
    return out


if __name__ == "__main__":
    r = analyse()
    (HERE / "gate_coverage.json").write_text(json.dumps(r, indent=2) + "\n")
    g = r["gap"]
    print(f"production steps: {g['production_steps']}")
    print(f"largest boltz2 target folded at production steps: "
          f"{g['largest_boltz2_target_folded_at_production_steps_aa']} aa")
    print(f"legs above it: {g['legs_above_that_size']} at {g['their_steps']} steps "
          f"({g['step_ratio']:.1f}x shorter)")
    print(f"incident at {r['incident']['size_aa']} aa / {r['incident']['steps']} steps covered: "
          f"{r['incident_in_the_gap']['covered']}")
