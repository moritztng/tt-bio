#!/usr/bin/env python3
"""Report, per vendored file, which upstream openfold3 release it came from.

Answers the only provenance question that matters for a training-equivalence claim:
for each file in ``tt_bio/_vendor/openfold3/``, is it upstream's file (modulo the
``tt_bio._vendor`` import rewrite), and if so whose? A file that matches no release
carries tt-bio changes and is listed with its diff size so it can be reviewed.

NOTICE records one version for the whole tree. That is true of almost all of it, but
"almost" is not a provenance record, so this prints the exceptions by name.

Usage:
    python scripts/of3_port/audit_vendor_provenance.py --dist <dir-of-unpacked-releases>

``--dist`` holds one subdirectory per release, named for its version, each containing
an ``openfold3/`` package dir (``pip download openfold3==X --no-deps`` + unzip).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VENDOR_NS = "tt_bio._vendor."


def unrewrite(text: str) -> str:
    return text.replace(VENDOR_NS + "openfold3.", "openfold3.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dist", type=Path, required=True)
    ap.add_argument(
        "--vendor",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tt_bio/_vendor/openfold3",
    )
    args = ap.parse_args()

    releases = sorted(
        (d for d in args.dist.iterdir() if (d / "openfold3").is_dir()),
        key=lambda d: [int(x) for x in re.findall(r"\d+", d.name)],
    )
    if not releases:
        raise SystemExit(f"no unpacked releases under {args.dist}")

    rows, local = [], []
    for f in sorted(args.vendor.rglob("*.py")):
        rel = f.relative_to(args.vendor).as_posix()
        mine = unrewrite(f.read_text())
        hits = [
            r.name
            for r in releases
            if (r / "openfold3" / rel).exists()
            and (r / "openfold3" / rel).read_text() == mine
        ]
        if hits:
            rows.append((rel, hits))
        else:
            present = [r for r in releases if (r / "openfold3" / rel).exists()]
            if not present:
                local.append((rel, "no upstream counterpart", 0))
            else:
                import difflib

                best, size = None, None
                for r in present:
                    n = sum(
                        1
                        for line in difflib.unified_diff(
                            (r / "openfold3" / rel).read_text().splitlines(),
                            mine.splitlines(),
                            n=0,
                        )
                        if line[:1] in "+-" and line[:3] not in ("+++", "---")
                    )
                    if size is None or n < size:
                        best, size = r.name, n
                local.append((rel, f"closest {best}", size))

    versions = [r.name for r in releases]
    print(f"vendored files: {len(rows) + len(local)}    releases compared: {', '.join(versions)}")
    print(f"exact upstream match: {len(rows)}    carrying tt-bio changes: {len(local)}\n")

    by_set: dict[tuple[str, ...], int] = {}
    for _, hits in rows:
        by_set[tuple(hits)] = by_set.get(tuple(hits), 0) + 1
    print("exact matches, grouped by which releases the file is identical in:")
    for hits, n in sorted(by_set.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d} file(s)  identical in {', '.join(hits)}")

    # A file identical in several releases is not evidence for any one of them. Only
    # files that single out one release pin the tree.
    print("\nfiles that match exactly ONE release (these pin the tree):")
    pinning = [(rel, hits[0]) for rel, hits in rows if len(hits) == 1]
    for rel, v in sorted(pinning, key=lambda t: t[1]):
        print(f"  {v:8s} {rel}")
    if not pinning:
        print("  (none)")

    print("\nfiles carrying tt-bio changes (diff lines vs closest release):")
    for rel, note, size in sorted(local, key=lambda t: -t[2]):
        print(f"  {size:5d}  {rel}   [{note}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
