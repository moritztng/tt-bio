"""Render the ladder JSONL rows into the campaign's evidence log.

One line per rung actually run, in the order they ran, with the numbers the classification
rests on. Reads only files the ladder wrote, so it cannot report a rung that never ran --
which is the failure mode a hand-maintained log has.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder import classify  # noqa: E402


def reclassify(r: dict) -> dict:
    """Name the wall from the saved log when the row does not carry one.

    A ladder already walking cannot be re-patched, and the first version of the classifier
    matched nothing because tt-metal wraps its refusal across two lines. The full stdout and
    stderr of every rung is on disk, so the classification is recoverable after the fact --
    which is the whole reason the driver saves it rather than a summary.
    """
    if r["verdict"] not in ("ERROR", "NO_STRUCTURE") or "request_bytes" in r:
        return r
    log = Path(r.get("log", ""))
    if not log.is_file():
        return r
    verdict, detail = classify(log.read_text(errors="replace"), r["rc"], False)
    if verdict.startswith("OOM"):
        return {**r, "verdict": verdict, **detail, "reclassified": True}
    return r


def rows(root: Path):
    for p in sorted(root.glob("ladder_*.jsonl")) + sorted(root.glob("guard/ladder_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                yield reclassify(json.loads(line))


def main() -> int:
    root = Path(sys.argv[1])
    out = []
    for r in sorted(rows(root), key=lambda r: (r["model"], r["rung"])):
        aa = r["rung"].split("_")[1]
        depth = r["rung"].split("_d")[-1] if "_d" in r["rung"] else "35"
        bits = [f"{r['ts']}  chip {r['device']}  {r['model']:<13} {aa:>5} aa  depth {depth:>5}",
                f"{r['verdict']:<12} {r['wall_s']:>8.1f} s"]
        if "request_bytes" in r:
            bits.append(f"refused {r['request_bytes']} B ({r['request_gib']} GiB) across "
                        f"{r['banks']} banks, {r['per_bank_mib']} MiB per bank")
            if "bank_size_mib" in r:
                bits.append(f"bank {r['bank_size_mib']} MiB, free {r['free_mib']} MiB, "
                            f"largest run {r['largest_free_mib']} MiB")
            bits.append(r.get("wall_kind", "UNCLASSIFIED"))
        if "structure" in r:
            bits.append(Path(r["structure"]).name)
        out.append("  ".join(bits))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
