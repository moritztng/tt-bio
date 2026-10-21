"""Merge the capture records into state/rfd4/goldens/INDEX.json (row rfd4-ref)."""

import json
from pathlib import Path

RECORDS = Path("/home/moritz/rfd4-goldens/records.jsonl")
OUT = Path("/home/moritz/.coworker/state/rfd4/goldens/INDEX.json")


def main() -> None:
    seen: dict[str, dict] = {}
    for line in RECORDS.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        seen[rec["path"]] = rec  # last write per file wins
    records = [r for r in seen.values() if Path(r["path"]).exists()]
    records.sort(key=lambda r: (r["feature"], r["cls"], r["module"], r["call"]))
    index = {
        "_note": "Upstream RFD4-Proteina goldens captured by rfd4-ref. Tensors live outside git at "
                 "/home/moritz/rfd4-goldens; this index carries the paths and hashes.",
        "upstream_sha": records[0]["upstream_sha"] if records else None,
        "env": "/home/moritz/rfd4-ref-env (CPU fp32, torch 2.14.1+cpu, python 3.12)",
        "count": len(records),
        "bytes": sum(r["bytes"] for r in records),
        "features": sorted({r["feature"] for r in records}),
        "modules": sorted({r["cls"] for r in records}),
        "seeds": sorted({r["seed"] for r in records}),
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(index, indent=1))
    print(f"{len(records)} captures, {index['bytes'] / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
