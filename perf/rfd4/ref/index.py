"""Merge the capture records into state/rfd4/goldens/INDEX.json (row rfd4-ref)."""

import hashlib
import json
from pathlib import Path

RECORDS = Path("/home/moritz/rfd4-goldens/records.jsonl")
OUT = Path("/home/moritz/.coworker/state/rfd4/goldens/INDEX.json")


PREP = Path("/home/moritz/rfd4-goldens/prep")


def prep_records(sha: str) -> list[dict]:
    """The annotated CIFs upstream's spec writer produces for each feature spec.

    These are the model's actual inputs: `rfd4-generate` reads the same file. The port's
    featurization is graded against them.
    """
    out = []
    for cif in sorted(PREP.glob("*.cif")):
        h = hashlib.sha256(cif.read_bytes()).hexdigest()
        out.append({
            "feature": cif.stem, "module": "scripts/write_condition_spec_cifs.py",
            "cls": "AnnotatedCIF", "call": 0, "seed": None, "upstream_sha": sha,
            "dtype": "text/cif", "device": "cpu", "path": str(cif),
            "bytes": cif.stat().st_size, "sha256": h,
            "in_args": None, "in_kwargs": None, "out": None,
            "f64": False, "f64_error": None, "overrides": [],
        })
    return out


def main() -> None:
    seen: dict[str, dict] = {}
    for line in RECORDS.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        seen[rec["path"]] = rec  # last write per file wins
    records = [r for r in seen.values() if Path(r["path"]).exists()]
    sha = records[0]["upstream_sha"] if records else None
    records += prep_records(sha)
    records.sort(key=lambda r: (r["feature"], r["cls"], r["module"], r["call"]))
    index = {
        "_note": "Upstream RFD4-Proteina goldens captured by rfd4-ref. Tensors live outside git at "
                 "/home/moritz/rfd4-goldens; this index carries the paths and hashes.",
        "upstream_sha": sha,
        "env": "/home/moritz/rfd4-ref-env (CPU fp32, torch 2.14.1+cpu, python 3.12)",
        "count": len(records),
        "bytes": sum(r["bytes"] for r in records),
        "features": sorted({r["feature"] for r in records}),
        "modules": sorted({r["cls"] for r in records}),
        "seeds": sorted({r["seed"] for r in records if r["seed"] is not None}),
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(index, indent=1))
    print(f"{len(records)} captures, {index['bytes'] / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
