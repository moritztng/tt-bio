#!/usr/bin/env python3
"""Assemble and (only on an explicit go) publish the AbAg-XM dataset.

Everything up to the upload is safe and repeatable: build the parquet tables, stage the
gzipped coordinates, copy the dataset card, and run the preflight checks. The upload itself
is gated behind --go, which exists so this cannot be run casually: **a public release needs
Moritz's explicit approval** and that is not something a script or an agent decides.

    python3 scripts/abag_xm_publish.py --out_dir ~/abag_xm/release            # assemble + check
    python3 scripts/abag_xm_publish.py --out_dir ~/abag_xm/release --go --repo ORG/abag-xm

Preflight, all blocking:
  * the dataset card has no unfilled {{placeholders}} -- a card with holes is worse than none
  * every generator has the same number of folds, so the slab is not silently lopsided
  * labels.parquet row count == folds x 50
  * every staged fold has 50 CIFs
`hf upload-large-folder`, never `hf upload`: ~49k files OOM-kills the latter.

The PUBLISHED: line in the state doc may only be written after the upload returns success
AND the dataset URL answers 200 -- this script prints that line only when it has verified
both, and never on the strength of having issued the command.
"""
import argparse
import json
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARD_SRC = ROOT / "docs" / "abag-xm-dataset-card.md"
SCRIPTS = ROOT / "scripts"


def run(cmd, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kw)


def preflight(out: Path, expect_samples: int) -> list[str]:
    fail = []
    card = out / "README.md"
    if not card.exists():
        fail.append("README.md (dataset card) missing")
    elif "{{" in card.read_text():
        holes = sorted({w for w in card.read_text().split() if "{{" in w})
        fail.append(f"dataset card has unfilled placeholders: {holes}")

    lab = out / "labels.parquet"
    if not lab.exists():
        fail.append("labels.parquet missing")
    else:
        import pandas as pd
        df = pd.read_parquet(lab)
        per_gen = Counter(df.generator)
        if len(set(per_gen.values())) > 1:
            fail.append(f"generators have unequal fold counts: {dict(per_gen)}")
        n_folds = df.groupby(["target", "generator"]).ngroups
        if len(df) != n_folds * expect_samples:
            fail.append(f"labels rows {len(df)} != {n_folds} folds x {expect_samples}")

    sdir = out / "structures"
    if not sdir.is_dir():
        fail.append("structures/ missing")
    else:
        short = [str(d.relative_to(sdir)) for d in sdir.glob("*/*")
                 if len(list(d.glob("*.cif.gz"))) != expect_samples]
        if short:
            fail.append(f"{len(short)} staged folds do not have {expect_samples} CIFs "
                        f"(e.g. {short[:3]})")
    return fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(Path.home() / "abag_xm" / "release"))
    ap.add_argument("--repo", help="HuggingFace dataset repo, ORG/NAME")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--go", action="store_true",
                    help="actually upload. Requires Moritz's explicit approval; without this "
                         "the script only assembles and checks.")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] parquet tables")
    run([sys.executable, str(SCRIPTS / "abag_xm_build_release_tables.py"), "--out_dir", str(out)])
    print("[2/4] coordinates")
    run([sys.executable, str(SCRIPTS / "abag_xm_stage_release.py"), "--out_dir", str(out)])
    print("[3/4] dataset card")
    if CARD_SRC.exists():
        (out / "README.md").write_text(CARD_SRC.read_text())
        print(f"  copied {CARD_SRC.name} -> README.md")

    print("[4/4] preflight")
    fail = preflight(out, a.samples)
    for f in fail:
        print(f"  FAIL: {f}")
    if fail:
        print(f"\n{len(fail)} preflight failure(s); not uploading.")
        return 1
    print("  all checks passed")

    if not a.go:
        print("\nAssembled and verified. Upload NOT attempted (no --go).")
        print("A public release needs Moritz's explicit approval.")
        return 0
    if not a.repo:
        print("--go requires --repo ORG/NAME", file=sys.stderr)
        return 2

    print(f"\nuploading to {a.repo}")
    r = run(["hf", "upload-large-folder", a.repo, str(out), "--repo-type", "dataset"])
    if r.returncode != 0:
        print(f"upload failed rc={r.returncode}; NOT published.")
        return 1
    url = f"https://huggingface.co/datasets/{a.repo}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            code = resp.status
    except Exception as e:
        print(f"upload returned success but {url} did not answer ({e}); NOT confirming.")
        return 1
    if code != 200:
        print(f"{url} returned {code}; NOT confirming.")
        return 1
    print(f"\nverified live. Record in the state doc:\n\nPUBLISHED: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
