#!/usr/bin/env python3
"""Grade a loss-parity or loss-curve artifact against the pre-registered G3 bars.

The bars live in ``tt_bio/train/parity.py`` and were committed before any RFD4 loss or curve
existed. This script only applies them, so the verdict is mechanical: it prints the exact
``LOSS-PARITY:`` / ``CURVE:`` line that ``state/r4t/rows/accuracy.md`` carries, and exits
non-zero on FAIL.

Loss parity (``--parity FILE``). JSON::

    {"dtype": "bfloat16",
     "ours":  [{"total": .., "bb_ca": .., ...}, ...],     # >= 8 batches
     "ref":   [{"total": .., "bb_ca": .., ...}, ...],     # upstream, same batches
     "provenance": {...}}                                  # free-form, printed back

Both sides must use upstream's term names verbatim: a term one side has and the other lacks is
a FAIL, because a loss that silently drops a term matches on everything it still computes.

``--floor perf/r4t/floor/up_bf16.json`` adds the registered bf16 floor (D37): a term is allowed
the larger of the relative bar and 1.5x upstream's OWN bf16 deviation on that term and batch,
because upstream itself misses 1e-2 on two of the nine. The floor is lined up by batch name and
the file's digest is checked, so a doc without names, an unknown batch, an edited floor or a
floor measured against a different reference is REFUSED (exit 2), never graded loosely.

Curve (``--curve FILE``). JSON::

    {"ours": [per-step loss, ...], "gpu_a": [...], "gpu_b": [...], "warmup": 1000}

``gpu_b`` is upstream re-run at a second seed on the same GPU and is not optional: the bar is
``max(2 % of upstream's final-200 mean, 1.5 x the seed-to-seed gap)``, so without it a
cross-arch gap has no scale to be judged against.

Curve logs (``--ours/--gpu-a/--gpu-b``). The JSONL a run writes through
``tt_bio.train.curve.CurveLog``, which carries the plan both sides executed. Preferred over
``--curve``: the plan digest in each header is what proves the two runs did the same work.
Exit codes: 0 PASS, 1 FAIL, 2 refused (no seed-B control, different plans, broken log).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_bio.train.curve import (CurveContractError, grade_curve_files,  # noqa: E402
                                read_curve, summarise)
from tt_bio.train.parity import (CURVE_MIN_STEPS, LOSS_PARITY_BARS,  # noqa: E402
                                 LOSS_PARITY_MIN_BATCHES, curve_match, floor_digest,
                                 loss_parity)


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _floor_column(doc: dict, path: str) -> list:
    """Line up the committed bf16 floor with this parity doc, by batch name. Refuses, never guesses.

    A positional match would silently pair batch 3's floor with batch 7's reading the first time
    an arm grades a different subset, and a looser floor reads as a pass.
    """
    fdoc = _load(path)
    rows = fdoc["rows"]
    want = floor_digest(rows)
    if fdoc.get("digest") != want:
        raise ValueError(f"floor file digest {fdoc.get('digest')!r} does not match its rows "
                         f"({want}): it was edited after it was committed")
    by_batch = {Path(str(r["batch"])).stem: r for r in rows}
    if len(by_batch) != len(rows):
        raise ValueError("floor file names a batch twice")
    pairs = (doc.get("provenance") or {}).get("pairs")
    if not pairs or len(pairs) != len(doc["ref"]):
        raise ValueError("parity doc carries no per-batch names, so the floor cannot be lined "
                         "up with it (provenance.pairs, one entry per reference batch)")
    out, ref_rows = [], doc["ref"]
    for i, p in enumerate(pairs):
        name = Path(str(p.get("batch", ""))).stem
        row = by_batch.get(name)
        if row is None:
            raise ValueError(f"batch {name!r} has no upstream bf16 floor in {path}")
        for k, v in row["ref"].items():
            got = ref_rows[i].get(k)
            if got is None:
                continue
            if abs(float(got) - float(v)) > 1e-9 * max(1.0, abs(float(v))):
                raise ValueError(f"batch {name!r} term {k!r}: the floor was measured against "
                                 f"ref {v!r}, this doc grades against {got!r}")
        out.append(row["up_bf16"])
    return out


def do_parity(doc: dict, floor_path: str | None = None) -> int:
    dtype = doc.get("dtype", "bfloat16")
    if dtype not in LOSS_PARITY_BARS:
        print(f"unknown dtype {dtype!r}; the bars are {sorted(LOSS_PARITY_BARS)}")
        return 2
    floor = None
    if floor_path:
        try:
            floor = _floor_column(doc, floor_path)
        except (KeyError, ValueError) as e:
            print(f"LOSS-PARITY: REFUSED -- {e}")
            return 2
    elif dtype == "bfloat16":
        print("  note: no --floor, so the bf16 arm is graded on the plain relative bar (D37 "
              "grades it against upstream's own bf16 deviation)")
    report = loss_parity(doc["ours"], doc["ref"], dtype, floor)
    print(report)
    if prov := doc.get("provenance"):
        print("  provenance: " + json.dumps(prov, sort_keys=True))
    if len(doc["ref"]) < LOSS_PARITY_MIN_BATCHES:
        print(f"  note: the bar needs n >= {LOSS_PARITY_MIN_BATCHES} batches")
    return 0 if report.passed else 1


def do_curve(doc: dict) -> int:
    report = curve_match(doc["ours"], doc["gpu_a"], doc["gpu_b"], int(doc["warmup"]))
    print(report)
    if prov := doc.get("provenance"):
        print("  provenance: " + json.dumps(prov, sort_keys=True))
    if len(doc["gpu_a"]) < CURVE_MIN_STEPS:
        print(f"  note: the bar needs >= {CURVE_MIN_STEPS} steps")
    return 0 if report.passed else 1


def do_logs(ours: str, gpu_a: str, gpu_b: str | None) -> int:
    """Grade three curve logs. A contract error is exit 2: refused, neither PASS nor FAIL."""
    try:
        report = grade_curve_files(ours, gpu_a, gpu_b)
    except CurveContractError as e:
        print(f"CURVE: REFUSED -- {e}")
        return 2
    print(report)
    for name, path in (("ours", ours), ("gpu A", gpu_a), ("gpu B", gpu_b)):
        header, losses = read_curve(path)
        prov = {k: header[k] for k in ("host", "sha", "upstream_sha", "board", "aiclk_mhz",
                                       "chip") if k in header}
        print(f"  {name}: {summarise(losses)} {json.dumps(prov, sort_keys=True)}")
    return 0 if report.passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--parity", metavar="FILE")
    g.add_argument("--curve", metavar="FILE")
    g.add_argument("--ours", metavar="LOG", help="curve log for our side (with --gpu-a/--gpu-b)")
    ap.add_argument("--gpu-a", metavar="LOG")
    ap.add_argument("--gpu-b", metavar="LOG")
    ap.add_argument("--floor", metavar="FILE",
                    help="committed upstream bf16 floor column (perf/r4t/floor/up_bf16.json)")
    a = ap.parse_args()
    if a.ours:
        if not a.gpu_a:
            print("CURVE: REFUSED -- --ours needs --gpu-a (and --gpu-b, the seed control)")
            return 2
        return do_logs(a.ours, a.gpu_a, a.gpu_b)
    if a.parity:
        return do_parity(_load(a.parity), a.floor)
    if a.floor:
        print("--floor applies to --parity only")
        return 2
    return do_curve(_load(a.curve))


if __name__ == "__main__":
    sys.exit(main())
