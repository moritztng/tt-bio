"""Emit the per-model verdict lines for FINDINGS.md from the ladder rows.

Typed numbers drift from measured ones, and this campaign's whole point is that a bare number
is indistinguishable from a guess. So the verdict line is generated: the largest rung MEASURED
to fold, the first rung measured to fail, and the class of that failure, all read from the
JSONL the ladder wrote.

The verdict is against the campaign's Wormhole bar of 1024 residues:

  PASS     folds the bar, and a first failure is on record, so the ceiling is BOUNDED.
  FAIL     does not fold the bar.
  PARTIAL  folds the bar, but nothing above it has failed yet: the bar is cleared and the
           ceiling is not bounded. PARTIAL is about the ceiling, never about the bar, and the
           line says so -- "PARTIAL" next to a model that folds 1536 would otherwise read as
           a problem with the model.

`pass_at` is the largest size below the FIRST failure, never merely the largest that folded --
the L1 clash class is not monotone in residue count (OpenDDE folds 544, throws at 576, folds
608), so publishing the largest passing rung would promise a size that throws.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import reclassify  # noqa: E402

BAR = 1024


def _size(rung: str) -> int:
    return int(rung.split("_")[1])


def _depth(r: dict) -> int | str:
    """The alignment depth this rung actually carried.

    Not a default: an affinity rung reads no alignment at all, and reporting it as the
    structure fixtures' 35 rows would say Nesso-1 was measured at a depth it never saw.
    """
    if r.get("command") == "affinity":
        return "none (affinity reads no alignment)"
    rung = r["rung"]
    return int(rung.split("_d")[-1]) if "_d" in rung else 35


def rows(root: Path):
    for p in sorted(root.glob("ladder_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = reclassify(json.loads(line))
                # A row whose log a later attempt overwrote cannot be classified, and a round-0
                # row that never ran the model (a guard refusal, a missing shared library) is
                # not a size measurement. Both are excluded by requiring a classified outcome.
                if r["verdict"] == "ERROR" and "request_bytes" not in r:
                    continue
                yield r


def main() -> int:
    root = Path(sys.argv[1])
    by = defaultdict(list)
    for r in rows(root):
        by[(r["model"], str(_depth(r)))].append(r)

    for (model, depth), rs in sorted(by.items()):
        rs.sort(key=lambda r: _size(r["rung"]))
        passed = [r for r in rs if r["verdict"] == "PASS"]
        failed = [r for r in rs if r["verdict"] != "PASS"]
        first_fail = failed[0] if failed else None
        below = [r for r in passed
                 if first_fail is None or _size(r["rung"]) < _size(first_fail["rung"])]
        if not below:
            verdict, cap = "FAIL", None
        else:
            cap = _size(below[-1]["rung"])
            verdict = "PASS" if cap >= BAR else "FAIL"
            if verdict == "PASS" and first_fail is None:
                verdict = "PARTIAL"
        cleared = "clears the 1024 bar" if verdict != "FAIL" else "does NOT clear the 1024 bar"
        bits = [f"{cleared}; {len(passed)}/{len(rs)} rungs folded at depth {depth}"]
        if cap:
            bits.append(f"largest below the first failure {cap} aa "
                        f"in {below[-1]['wall_s']:.0f} s")
        if first_fail:
            k = first_fail.get("wall_kind", "UNCLASSIFIED")
            b = first_fail.get("request_bytes")
            where = f" on {b} B" if b else ""
            extra = ""
            if "largest_free_mib" in first_fail:
                extra = (f" ({first_fail['per_bank_mib']} MiB wanted per bank, bank "
                         f"{first_fail['bank_size_mib']} MiB, free {first_fail['free_mib']} MiB, "
                         f"largest run {first_fail['largest_free_mib']} MiB)")
            bits.append(f"first failure {_size(first_fail['rung'])} aa, "
                        f"{first_fail['verdict']}{where}{extra}; cause {k}")
        else:
            bits.append("no failure found, so this number is a floor and not a ceiling: "
                        "the limit reached is the top of the ladder, not the chip")
        if first_fail is None:
            cause = ("no rung above " + str(cap) + " has been run, so the limit here is the top "
                     "of the ladder and not the chip; the ceiling is unbounded, not reached")
        elif first_fail.get("wall_kind") == "ONE_OVERSIZED_TENSOR":
            cause = (f"one oversized tensor: the per-bank share of the "
                     f"{first_fail['request_bytes']} B request does not fit an empty bank")
        elif first_fail.get("wall_kind") == "CUMULATIVE_RESIDENCY":
            cause = (f"cumulative residency: {first_fail.get('free_mib')} MiB free per bank "
                     f"against {first_fail.get('per_bank_mib')} MiB wanted, chip "
                     f"{first_fail.get('occupancy_pct')}% full")
        elif str(first_fail.get("wall_kind", "")).startswith("FRAGMENTATION"):
            cause = (f"fragmentation: {first_fail.get('free_mib')} MiB free per bank against "
                     f"{first_fail.get('per_bank_mib')} MiB wanted, but the largest free block "
                     f"is {first_fail.get('largest_free_mib')} MiB"
                     + (", and the chip is "
                        f"{first_fail.get('occupancy_pct')}% full so compaction alone cannot "
                        "reach it"
                        if first_fail.get("wall_kind") == "FRAGMENTATION_ON_FULL_CHIP" else ""))
        else:
            cause = (f"{first_fail['verdict']}, wall kind "
                     f"{first_fail.get('wall_kind', 'UNCLASSIFIED')}")
        if any(r.get("env") for r in rs):
            envs = sorted({k for r in rs for k in (r.get("env") or {})
                           if k != "TT_BIO_SIZE_LIMIT"})
            if envs:
                bits.append("some rungs ran with " + ",".join(envs))
        print(f"MODEL {model}: {verdict} " + "; ".join(bits))
        # A SECOND line, deliberately, and not because the first has no room. The campaign's
        # DONE_CHECK looks for a stated cause on a line AFTER the verdict line, and a reader
        # skimming a column of verdicts wants the one-clause reason next to it rather than at
        # the end of a long sentence. Same fact, said where both of them look.
        print(f"    cause: {cause}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
