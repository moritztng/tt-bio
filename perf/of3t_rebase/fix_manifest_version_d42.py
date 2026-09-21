#!/usr/bin/env python3
"""D42: re-emit the manifests whose provenance field names the revision D23 disqualified.

`bundle_min.py` used to write

    "openfold3": getattr(openfold3, "__version__", "0.5.0 (git checkout)")

and the 0.4.3 checkout does not define `__version__`, so the fallback fired on every build and
asserted **0.5.0** into the manifest of the file that certifies this campaign's reference. The
build was right -- PKG-INFO says 0.4.3, the model has 4,170 parameters where 0.5.0 gives 4,147,
and the checkpoint loads missing=1 / unexpected=0 -- but an auditor reading the manifest sees the
disqualified revision and stops, and they would be right to stop.

The generator is fixed (`_openfold3_version`, which reads the tree's own metadata and returns
"unknown" rather than guessing). This re-emits the manifests that were already written, because
the payload is untouched -- a metadata write, not a rebuild.

Two places, and they are not the same shape:
  * `run/out_043_fd/manifest.json`      -> `versions.openfold3`      (bundle_min's own schema)
  * `bundle_min_043/MANIFEST.json`      -> `run_record.versions.openfold3`, the embedded run
    record. That file ALSO carries a correct `upstream.version: "0.4.3"`, so until now it
    contradicted itself.

Refuses to write unless it finds exactly the wrong string, so it cannot silently no-op against a
file someone already fixed, and it re-checks the payload hash so a metadata write can never be
confused with a rebuild.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

WRONG = "0.5.0 (git checkout)"


def sha256(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def patch(path: Path, dotted: str, value: str, apply: bool) -> bool:
    d = json.loads(path.read_text())
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node.get(k) if isinstance(node, dict) else None
        if node is None:
            print(f"  {path}: no {dotted} -- nothing to do")
            return False
    cur = node.get(keys[-1])
    if cur == value:
        print(f"  {path}: {dotted} already {value!r}")
        return False
    if cur != WRONG:
        print(f"  {path}: {dotted} is {cur!r}, not the known-wrong string -- REFUSING to write")
        return False
    node[keys[-1]] = value
    d.setdefault("provenance_corrections", []).append({
        "defect": "D42",
        "field": dotted,
        "was": WRONG,
        "now": value,
        "why": ("the 0.4.3 tree defines no openfold3.__version__, so bundle_min.py's getattr "
                "fallback asserted a specific WRONG revision instead of admitting ignorance; "
                "the generator now reads the tree's own PKG-INFO and returns 'unknown' when it "
                "cannot tell"),
        "payload_rebuilt": False,
    })
    print(f"  {path}: {dotted}  {WRONG!r} -> {value!r}")
    if apply:
        path.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/home/ttuser/of3t_rebase/bundle_min_043")
    ap.add_argument("--fd-run", default="/home/ttuser/of3t_rebase/run/out_043_fd")
    ap.add_argument("--version", required=True, help="measured, e.g. '0.4.3 (from PKG-INFO ...)'")
    ap.add_argument("--also", nargs="*", default=[], metavar="PATH:DOTTED",
                    help="extra manifests to correct, e.g. the IN-GIT copy. Amendment 26 named "
                         "the two on-disk files and missed "
                         "perf/of3t_rebase/refbuild/MANIFEST_043.json, which is the copy an "
                         "auditor actually reads out of the repository.")
    ap.add_argument("--no-payload-check", action="store_true",
                    help="for corrections made where the payload is not alongside, e.g. in-git")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    b, f = Path(a.bundle), Path(a.fd_run)
    payload = b / "grads_f64_043.pt"
    before = None if a.no_payload_check else sha256(payload)
    if before:
        print(f"payload {payload.name} sha256 before: {before}")

    n = 0
    if not a.no_payload_check:
        n += patch(f / "manifest.json", "versions.openfold3", a.version, a.apply)
        n += patch(b / "MANIFEST.json", "run_record.versions.openfold3", a.version, a.apply)
    for spec in a.also:
        path, _, dotted = spec.rpartition(":")
        n += patch(Path(path), dotted, a.version, a.apply)

    if before:
        after = sha256(payload)
        print(f"payload {payload.name} sha256 after : {after}")
        if before != after:
            raise SystemExit("PAYLOAD CHANGED -- this is supposed to be a metadata write only")
    print(f"{'wrote' if a.apply else 'would write'} {n} manifest(s); payload unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
