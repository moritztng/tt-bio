"""Every `[served, declined]` counter in tt_bio must be reachable from the lever census.

`scripts/lever_census.py` exists to answer the half of the question a default cannot: a lever
that is merged and switched on still delivers nothing if its guard never admits a real fold.
Its `LEVERS` table is hand-maintained, and on 2026-09-14 it referenced 21 of the 37 counters
that existed -- so the gate built to catch a dark lever was itself blind to 17 levers,
including both the ROOF campaign had landed on main and the conditioning lever main's accuracy
position rests on. A hand-maintained list that silently misses new entries is a recurring
defect class here, so this test converts the gap into a failure.

Adding a counter without a census row fails this test. If a counter genuinely should not be
censused, put it in EXEMPT with the reason -- an explicit line, not an omission.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "tt_bio"
CENSUS = ROOT / "scripts" / "lever_census.py"

#: counter -> why it is deliberately outside the census.
EXEMPT = {
    # Design-model levers. The census runs one fold per model and these models are not in its
    # matrix; they keep their counters for their own task-local A/Bs.
    "tt_bio.rfd3.model.ATOM_PAIR_BLOCK_STATS": "rfd3, not in the census fold matrix",
    "tt_bio.rfd3.model.PTL1STATS": "rfd3, not in the census fold matrix",
    "tt_bio.rfd3.model.PZSTATS": "rfd3, not in the census fold matrix",
    "tt_bio.rfd3_bias.DSTATS": "rfd3, not in the census fold matrix",
    "tt_bio.protenix.RELP_STATS": "protenix-v2 relpos, covered by that model's own ladder",
    # Path censuses, not lever guards: they record which path a call took (whole / blocked /
    # narrowed) and are driven by byte budgets, not by an env flag, so there is no resolved value
    # for the census to report. Read them from a fold's own dump instead.
    "tt_bio.tenstorrent.OPM_ROW_STATS": "path census driven by OPM_Z_BUDGET_BYTES, no flag",
    "tt_bio.tenstorrent.PWA_DEPTH_STATS": "path census driven by PWA_DEPTH_BUDGET_BYTES, no flag",
    # Not a lever: a fallback counter, where served>0 is the bad direction.
    "tt_bio.esmc.WINDOW_FALLBACK_STATS": "fallback counter, not a lever guard",
}


def _declared_counters() -> dict[str, Path]:
    """Every module-level lever counter under tt_bio, as a dotted path.

    Two shapes are in use and the census handles both: a `[served, declined]` list (`how="stats"`)
    and a keyed dict (`how="stats-dict"`, e.g. `FP32_SOFTMAX_STATS`). Matching only the list shape
    made the first draft of this test report the dict one as dangling, which is a false alarm on a
    correct codebase -- worse than no test.
    """
    found: dict[str, Path] = {}
    for path in sorted(PKG.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        mod = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        for node in tree.body:            # module level only -- a local is not a census target
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if not isinstance(tgt, ast.Name) or "STATS" not in tgt.id:
                    continue
                v = node.value
                is_pair = (isinstance(v, ast.List) and len(v.elts) == 2
                           and all(isinstance(e, ast.Constant) and e.value == 0 for e in v.elts))
                is_dict = isinstance(v, ast.Dict) and bool(v.keys)
                if is_pair or is_dict:
                    found[f"{mod}.{tgt.id}"] = path
    return found


def _censused_counters() -> set[str]:
    return set(re.findall(r'"(tt_bio\.[\w.]+STATS[\w]*)"', CENSUS.read_text()))


def test_every_counter_is_censused_or_exempt():
    declared = _declared_counters()
    assert declared, "found no [0, 0] counters at all -- the scanner is broken, not the package"
    missing = sorted(set(declared) - _censused_counters() - set(EXEMPT))
    assert not missing, (
        "these [served, declined] counters have no row in scripts/lever_census.py and no EXEMPT "
        "entry, so the lever census cannot tell whether they ever fire:\n  "
        + "\n  ".join(f"{m}  ({declared[m].relative_to(ROOT)})" for m in missing)
        + "\nAdd a LEVERS row, or an EXEMPT entry stating why it is not a lever."
    )


def test_exemptions_still_exist():
    """A stale exemption hides a counter that was renamed rather than one that is exempt."""
    declared = set(_declared_counters())
    stale = sorted(set(EXEMPT) - declared)
    assert not stale, f"EXEMPT names counters that no longer exist: {stale}"


def test_censused_counters_all_exist():
    """A census row pointing at a counter that is gone reads as served=0 forever."""
    declared = set(_declared_counters())
    dangling = sorted(_censused_counters() - declared)
    assert not dangling, (
        "scripts/lever_census.py references counters that do not exist; each would read "
        f"served=0 forever and look like a dark lever: {dangling}")
