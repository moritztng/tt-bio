"""Does a re-run reproduce a committed baseline to the last digit, and is every CA cloud readable?

Step 6B's acceptance check is not a tolerance, it is equality: a fresh process in a different order
must return the committed scalars exactly, because that is what proves `_template_key` holds and
that the host arm carries no position dependence. This compares banked shard rows against the
committed file id by id and scalar by scalar, and reports the ids a baseline does not cover rather
than quietly scoring only the intersection.

`--ca` also loads every CA cloud the same run wrote. qb1 was hard power-cycled on 2026-08-21 four
minutes after the last shard write, so a banked row whose `.npy` was still in the page cache would
resume as present-and-complete while being truncated on disk. The resume path trusts existence.

    PYTHONPATH=. python3 scripts/af2_port/reproduce_check.py \\
        --banked '.af2ig_p16/designpop_pxd196/host_complex_shard*.jsonl' \\
        --baseline scripts/af2_port/parity_artifacts/designpop_pxd196/scores_host.jsonl \\
        --ca .af2ig_p16/designpop_pxd196/ca_reference
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def rows(paths: list[str]) -> dict:
    out = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["id"] not in out, "%s duplicated across shards" % row["id"]
            out[row["id"]] = row
    return out


def scalars(row: dict) -> dict:
    """Every number the filter reads, flattened. `dev` is derived from `ref` but is what the
    decision uses, so a mismatch in either is a mismatch."""
    flat = dict(row.get("ref", {}))
    flat.update({"dev_" + k: v for k, v in row.get("dev", {}).items()})
    return flat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banked", required=True, help="glob of the re-run's shard jsonl files")
    ap.add_argument("--baseline", default=None, help="the committed file the re-run must reproduce")
    ap.add_argument("--ca", default=None, help="CA directory to load every cloud from")
    ap.add_argument("--stage", default="complex", choices=["complex", "monomer"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    got = rows(sorted(glob.glob(a.banked)))
    want = rows([a.baseline]) if a.baseline else {}
    report = {"banked": len(got), "baseline": len(want), "stage": a.stage}

    same, differ = [], []
    for rid in sorted(set(got) & set(want)):
        g, w = scalars(got[rid]), scalars(want[rid])
        keys = sorted(set(g) & set(w))
        assert keys, "%s: no comparable scalars between the two files" % rid
        bad = {k: [w[k], g[k]] for k in keys if g[k] != w[k]}
        (differ.append({"id": rid, "scalars": bad}) if bad else same.append(rid))
    report.update({
        "compared": len(same) + len(differ),
        "scalars_per_row": len(set(scalars(got[same[0]])) & set(scalars(want[same[0]])))
                           if same else 0,
        "bit_identical": len(same),
        "mismatched": differ,
        "banked_not_in_baseline": sorted(set(got) - set(want)),
        "baseline_not_yet_rerun": sorted(set(want) - set(got)),
    })

    if a.ca:
        import numpy as np
        ca_dir, bad_ca, loaded = Path(a.ca), [], 0
        for rid in sorted(got):
            path = ca_dir / ("%s.%s_ca.npy" % (rid, a.stage))
            if not path.exists():
                bad_ca.append({"id": rid, "why": "missing"})
                continue
            try:
                cloud = np.load(path)
            except Exception as exc:                      # a truncated .npy raises here
                bad_ca.append({"id": rid, "why": repr(exc)})
                continue
            if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
                bad_ca.append({"id": rid, "why": "shape %s finite=%s"
                                              % (cloud.shape, bool(np.isfinite(cloud).all()))})
                continue
            if cloud.shape[0] != got[rid].get("tokens_scored", cloud.shape[0]):
                bad_ca.append({"id": rid, "why": "%d rows, row says tokens_scored=%d"
                                              % (cloud.shape[0], got[rid]["tokens_scored"])})
                continue
            loaded += 1
        report["ca"] = {"loaded": loaded, "bad": bad_ca}

    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    ok = not differ and not report.get("ca", {}).get("bad")
    print("REPRODUCE %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
