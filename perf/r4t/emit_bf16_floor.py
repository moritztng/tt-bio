#!/usr/bin/env python3
"""Turn a ``bf16_floor.py`` log into the committed floor column the G3 bf16 bar divides by (D37).

The bar for a bf16 term is ``max(1e-2 |ref|, 1.5 |upstream_bf16 - ref|)``: a TT reading is only a
finding if it is bigger than what the SAME model in bf16 does to that term with nothing ported.
That second half is data, so it is committed with a digest and the grader refuses a floor file
whose rows do not hash to it.

    emit_bf16_floor.py ~/r4t/accuracy/bf16floor/bf16.log --out perf/r4t/floor/up_bf16.json \
        --sha <tt-bio sha> --upstream-sha <rfd4 sha>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_bio.train.parity import floor_digest  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sha", default="")
    ap.add_argument("--upstream-sha", default="")
    ap.add_argument("--weights", default="nvidia/RFD4-Proteina@a06e2509")
    a = ap.parse_args(argv)

    rows, summary = [], None
    for ln in a.log.read_text(errors="replace").splitlines():
        if not ln.startswith("{"):
            continue
        rec = json.loads(ln)
        if rec.get("summary"):
            summary = rec
            continue
        if rec.get("arm") != "bf16":
            continue
        rows.append(dict(batch=rec["batch"], up_bf16=rec["ours"], ref=rec["ref"]))
    if not rows:
        print("no bf16 rows in the log")
        return 1
    if summary and summary.get("batches") != len(rows):
        print(f"log summary says {summary['batches']} batches, read {len(rows)}")
        return 1
    doc = dict(
        rows=rows,
        digest=floor_digest(rows),
        provenance=dict(
            arm="bf16", host=platform.node(), sha=a.sha, upstream_sha=a.upstream_sha,
            weights=a.weights, device="cpu", autocast="torch.autocast('cpu', bfloat16)",
            log=str(a.log), script="perf/r4t/bf16_floor.py",
            note="fp32 control arm on the same harness lands at the replay floor (2.5e-6)"),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"{len(rows)} batches -> {a.out}  digest {doc['digest']}")
    print("  sha256(log) " + hashlib.sha256(a.log.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
