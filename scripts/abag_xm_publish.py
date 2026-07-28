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
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARD_SRC = ROOT / "docs" / "abag-xm-dataset-card.md"
# Assembly runs without --repo, and the card still has to name one.
# This is the target the frozen design records; --repo overrides it.
DEFAULT_REPO = "tt-moritz/abag-xm"
TARGETS_SRC = ROOT / "docs" / "implementation-parity-data" / "abag-xm-targets.parquet"
SCRIPTS = ROOT / "scripts"


def run(cmd, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kw)


def _tt_bio_url():
    """The tt-bio URL the card links to, derived from this checkout's own remote.

    Derived rather than hardcoded so it cannot drift from where the code actually lives.
    """
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                           capture_output=True, text=True, timeout=15)
        url = r.stdout.strip()
    except Exception:
        url = ""
    if not url:
        return "https://github.com/moritztng/tt-bio"
    if url.startswith("git@"):                       # git@github.com:owner/repo.git
        url = "https://" + url[4:].replace(":", "/", 1)
    return url[:-4] if url.endswith(".git") else url


def _fill_card(text, out: Path, repo):
    """Substitute the card's {{placeholders}} from the tables just built.

    The counts come from labels.parquet, never from a constant: the card states the size of the
    release, and a hand-typed size is a claim about data rather than a description of it. Before
    this, nothing filled these at all -- the preflight only detected them -- so a real publish
    would have blocked until someone edited the card by hand.
    """
    vals = {"HF_REPO": repo or DEFAULT_REPO, "TT_BIO_URL": _tt_bio_url()}
    lab = out / "labels.parquet"
    if lab.exists():
        import pandas as pd
        df = pd.read_parquet(lab)
        vals["N_TARGETS"] = str(df.target.nunique())
        vals["N_SAMPLES_TOTAL"] = f"{len(df):,}"
    filled = {}
    for k, v in vals.items():
        token = "{{%s}}" % k
        if token in text:
            text = text.replace(token, v)
            filled[k] = v
    missing = {w for w in text.split() if "{{" in w}
    return text, filled, missing


def preflight(out: Path, expect_samples: int, expect_targets: int = 164) -> list[str]:
    fail = []
    card = out / "README.md"
    if not card.exists():
        fail.append("README.md (dataset card) missing")
    elif "{{" in card.read_text():
        holes = sorted({w for w in card.read_text().split() if "{{" in w})
        fail.append(f"dataset card has unfilled placeholders: {holes}")

    for name in ("targets.parquet", "labels.parquet", "ensembles.parquet"):
        if not (out / name).exists():
            fail.append(f"{name} missing -- the dataset card documents all three tables")

    lab = out / "labels.parquet"
    if not lab.exists():
        fail.append("labels.parquet missing")
    else:
        import pandas as pd
        df = pd.read_parquet(lab)
        per_gen = Counter(df.generator)
        if len(set(per_gen.values())) > 1:
            fail.append(f"generators have unequal fold counts: {dict(per_gen)}")

        # Balanced is not the same as complete, and the existing checks only test balance. A slab with
        # 100 of 164 targets folded by all three generators is internally consistent: generator counts
        # match, rows == folds x samples, targets.parquet covers what is present. It would pass every
        # check above and publish as if it were the dataset the card describes.
        #
        # So assert the target count too. --expect_targets 0 releases a deliberate subset, which then
        # has to be an explicit decision rather than an accident.
        n_targets = df.target.nunique()
        if expect_targets and n_targets != expect_targets:
            fail.append(
                f"labels.parquet covers {n_targets} targets, expected {expect_targets} -- the slab is "
                f"INCOMPLETE, not merely unbalanced. Pass --expect_targets 0 to publish a subset on "
                f"purpose.")
        n_folds = df.groupby(["target", "generator"]).ngroups
        if len(df) != n_folds * expect_samples:
            fail.append(f"labels rows {len(df)} != {n_folds} folds x {expect_samples}")
        # The card promises every row traces to the exact code that produced it. A "-dirty"
        # commit means the worktree had uncommitted changes when that fold ran, so it traces to
        # nothing reproducible -- regenerate those folds rather than publish an untraceable row.
        dirty = df[df.tt_bio_commit.astype(str).str.endswith("-dirty")]
        if len(dirty):
            folds = sorted({f"{t}/{g}" for t, g in zip(dirty.target, dirty.generator)})
            fail.append(f"{len(dirty)} rows ({len(folds)} folds) ran from a DIRTY worktree and "
                        f"cannot be traced to a commit: {folds[:6]}"
                        + (" ..." if len(folds) > 6 else ""))

        # Two of the tools this campaign uses may not be redistributed, and the release is
        # CC-BY-4.0: ABAG-Rank's weights and bundled examples are CC BY-NC 4.0, and PSBench's
        # models are AF3 outputs we do not own. Both were only ever meant to be internal scoring
        # tools. They are excluded structurally today -- labels.parquet is built from the per-fold
        # label JSONs and never from ranker_scores.csv -- but "structurally" is one refactor away
        # from "accidentally", and a licence breach is not the kind of thing to notice after
        # upload. Assert it instead of trusting the data flow.
        banned = {c: t for c in df.columns
                  for t, pats in (("ABAG-Rank (weights CC BY-NC 4.0)", ("abag_rank", "abagrank")),
                                  ("PSBench (AF3 outputs, not ours)", ("psbench", "dockq_wave")))
                  if any(p in c.lower() for p in pats)}
        if banned:
            fail.append("labels.parquet carries non-redistributable columns -- "
                        + "; ".join(f"{c!r} from {t}" for c, t in sorted(banned.items())))

        tgt = out / "targets.parquet"
        if tgt.exists():
            # A targets table that does not cover every released target leaves rows with no
            # sequence or provenance to join against -- worse than an obviously missing file.
            have = set(pd.read_parquet(tgt).pdb_id)
            orphan = sorted(set(df.target) - have)
            if orphan:
                fail.append(f"{len(orphan)} released targets absent from targets.parquet "
                            f"(e.g. {orphan[:3]})")

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
    ap.add_argument("--expect_targets", type=int, default=164,
                    help="target count the release must cover (0 = allow a subset, "
                         "deliberately)")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--go", action="store_true",
                    help="actually upload. Requires Moritz's explicit approval; without this "
                         "the script only assembles and checks.")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] parquet tables")
    run([sys.executable, str(SCRIPTS / "abag_xm_build_release_tables.py"), "--out_dir", str(out)])
    # targets.parquet is the third table the card documents, and nothing was producing it: a
    # release would have shipped a dataset whose own card describes a file that is not there.
    # It is a Phase-1 artifact already committed to the repo -- copied, never rebuilt here,
    # because abag_xm_build_manifest.py re-fetches every mmCIF from RCSB.
    if TARGETS_SRC.exists():
        shutil.copyfile(TARGETS_SRC, out / "targets.parquet")
        print(f"  copied {TARGETS_SRC.name} -> targets.parquet")
    else:
        print(f"  !! {TARGETS_SRC} missing -- targets.parquet will not be in the release")

    print("[2/4] coordinates")
    run([sys.executable, str(SCRIPTS / "abag_xm_stage_release.py"), "--out_dir", str(out)])
    print("[3/4] dataset card")
    if CARD_SRC.exists():
        text, filled, missing = _fill_card(CARD_SRC.read_text(), out, a.repo)
        (out / "README.md").write_text(text)
        print(f"  copied {CARD_SRC.name} -> README.md")
        for k, v in sorted(filled.items()):
            print(f"    {k} = {v}")
        if missing:
            print(f"    !! still unfilled: {sorted(missing)}")

    print("[4/4] preflight")
    fail = preflight(out, a.samples, a.expect_targets)
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
