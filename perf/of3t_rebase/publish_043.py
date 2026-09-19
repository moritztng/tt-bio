#!/usr/bin/env python3
"""Assemble and publish BUNDLE-MIN-043: the reference bundle rebuilt at upstream 0.4.3.

Same batch, same replayed draws, same r = 0, same float64 as the published 0.5.0 bundle. The
only thing that moves between the two artifacts is the upstream revision, which is the point.

What a consumer gets: the payload under `--dst`, a MANIFEST.json in the same schema the 0.5.0
bundle publishes so `capture_trunk_boundary.py --manifest-json` can hash against it, and the
A13 reproduction result stating whether run A and run B agree bit for bit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", type=Path, required=True)
    ap.add_argument("--run-b", type=Path, default=None)
    ap.add_argument("--src-bundle", type=Path, default=Path("/home/ttuser/of3t/bundle_min"),
                    help="where the carried-in batch and draws live; they are not rebuilt")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--gate", type=Path, required=True, help="refbuild_keygate.py JSON at 0.4.3")
    ap.add_argument("--gate-control", type=Path, required=True, help="the same at 0.5.0")
    ap.add_argument("--a13", type=Path, default=None, help="compare_grads.py JSON, run A vs B")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    a.dst.mkdir(parents=True, exist_ok=True)
    gate = json.loads(a.gate.read_text())
    ctl = json.loads(a.gate_control.read_text())
    if gate["gate1_keys"]["unexpected"]["n"]:
        raise SystemExit("the 0.4.3 gate did not pass; there is nothing here to publish")

    rec = json.loads((a.run_a / "run_record.json").read_text())

    payload = {
        "grads_f64_043.pt": a.run_a / "grads_f64.pt",
        "grad_presence_043.json": a.run_a / "grad_presence.json",
        "w0_043.pt": a.run_a / "w0.pt",
        "batch_step003.pt": a.src_bundle / "batch_step003.pt",
        "draws_recycles0.pt": a.src_bundle / "draws_recycles0.pt",
    }
    artifacts = []
    for name, src in payload.items():
        dst = a.dst / name
        if not dst.exists():
            # hardlink where we can, copy across filesystems. The bytes are the artifact; a
            # second copy of 4.5 GB buys nothing.
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        artifacts.append({"file": name, "bytes": dst.stat().st_size,
                          "sha256": sha256_file(dst),
                          "carried_in": src.parent == a.src_bundle})

    a13 = json.loads(a.a13.read_text()) if a.a13 and a.a13.is_file() else None

    man = {
        "bundle": "BUNDLE-MIN-043",
        "published_by": "of3t-rebase",
        "what_it_is":
            "The OF3T reference gradient rebuilt at the upstream revision of3-p2-155k.pt belongs "
            "to. D23/R126: the 0.5.0 bundle it replaces ran a checkpoint 0.5.0's own registry "
            "marks '>=0.4,<0.4.4dev0' and lists in LEGACY_CHECKPOINTS, dropping 48 "
            "layer_norm_z.weight tensors through strict=False.",
        "upstream": {
            "version": gate["tree_version"],
            "tree": gate["tree"],
            "source": "openfold3 sdist, extracted; properties verified in the tree used",
            "properties": gate["tree_properties"],
        },
        "key_gate": {
            "rule": gate["gate1_keys"]["upstream_rule"],
            "this_build": {"missing": gate["gate1_keys"]["missing"]["n"],
                           "unexpected": gate["gate1_keys"]["unexpected"]["n"],
                           "missing_keys": gate["gate1_keys"]["missing"]["all"],
                           "upstream_branch": gate["gate1_keys"]["upstream_branch_taken"],
                           "upstream_called": gate["gate1_keys"]["upstream_called"]},
            "negative_control_0_5_0": {
                "missing": ctl["gate1_keys"]["missing"]["n"],
                "unexpected": ctl["gate1_keys"]["unexpected"]["n"],
                "unexpected_by_leaf": ctl["gate1_keys"]["unexpected"]["by_leaf"],
                "upstream_branch": ctl["gate1_keys"]["upstream_branch_taken"],
                "upstream_called": ctl["gate1_keys"]["upstream_called"]},
            "version_window": {"this_build": gate["gate2_version_window"],
                               "negative_control": ctl["gate2_version_window"]},
            "enforced_in": "perf/of3t_reference/bundle_min.py refuses a non-empty unexpected set",
        },
        "artifacts": artifacts,
        "checkpoint": gate["checkpoint"],
        "batch": {"pdb_id": "5nw3", "n_tokens": 56,
                  "note": "carried in by hash from the 0.5.0 bundle and NOT rebuilt: "
                          "featurisation is deterministic within one environment and not "
                          "across one, so rebuilding it would move the batch as well as the "
                          "revision and the comparison would have two moving parts."},
        "run_record": rec,
        "determinism_A13": a13,
        "canonical_location": {"host": "qb2", "path": str(a.dst)},
        "consumer_rules": [
            "Use grads_f64_043.pt for anything compared against a 0.4.3 reference, and "
            "grads_f64_r0.pt only for restating what the 0.5.0 bundle read.",
            "The two bundles are NOT interchangeable and their difference is not noise.",
            "Match draws_recycles0.pt: both bundles replayed the identical draws, so a "
            "difference between them is the revision and nothing else.",
            "None is not zero. Compare the presence pattern as a set before any magnitude.",
        ],
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
    }
    (a.dst / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(man, indent=1) + "\n")
    for x in artifacts:
        print(f"{x['sha256']}  {x['file']}  ({x['bytes']} bytes"
              f"{', carried in' if x['carried_in'] else ''})")
    print(f"key gate: missing={man['key_gate']['this_build']['missing']} "
          f"unexpected={man['key_gate']['this_build']['unexpected']}")
    if a13:
        print(f"A13: {json.dumps({k: v for k, v in a13.items() if not isinstance(v, list)})}")
    print("wrote", a.dst / "MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
