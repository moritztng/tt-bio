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


YAML_DIR = ROOT / "examples" / "abag_xm"

# The YAML chain ids this campaign folds: A antigen, H heavy, L light.
FOLD_SEQ_COLS = {"A": "fold_seq_antigen", "H": "fold_seq_heavy", "L": "fold_seq_light"}


def _add_fold_sequences(targets_parquet):
    """Add each target's folded chain sequences to the released targets.parquet.

    The card's file table says targets.parquet carries "sequences". It did not: the committed
    manifest has cdrh3_sequences and fold_resolved_seq_length_1/2, i.e. one CDR loop and two
    integers. So a downloader could not reconstruct what was actually folded, could not map a
    model chain to a YAML chain, and could not re-run a fold without going back to the PDB --
    for a dataset whose whole premise is comparing predictions of a known input.

    Taken from the fold YAMLs, which ARE the model input, rather than from the mmCIF: the
    deposited sequence is not always what was folded (constructs differ, and the mmCIF carries
    modified residues the YAML declares as standard ones). Added here rather than in
    abag_xm_build_manifest.py because that script re-fetches all 164 mmCIFs from RCSB, and
    regenerating a committed Phase-1 artifact to append derived columns is the wrong trade.
    """
    import pandas as pd
    import yaml as _yaml

    df = pd.read_parquet(targets_parquet)
    got = {c: [] for c in FOLD_SEQ_COLS.values()}
    missing = []
    for pdb in df.pdb_id:
        path = YAML_DIR / f"{pdb}.yaml"
        seqs = {}
        if path.exists():
            try:
                doc = _yaml.safe_load(path.read_text())
                for entry in doc.get("sequences", []):
                    prot = entry.get("protein") or {}
                    if prot.get("id") in FOLD_SEQ_COLS:
                        seqs[prot["id"]] = prot.get("sequence")
            except Exception:
                pass
        if not seqs:
            missing.append(pdb)
        for cid, col in FOLD_SEQ_COLS.items():
            got[col].append(seqs.get(cid))
    for col, vals in got.items():
        df[col] = vals
    df.to_parquet(targets_parquet, index=False)
    have = int(df[FOLD_SEQ_COLS["A"]].notna().sum())
    print(f"  + fold sequences: antigen {have}/{len(df)}, "
          f"heavy {int(df[FOLD_SEQ_COLS['H']].notna().sum())}/{len(df)}, "
          f"light {int(df[FOLD_SEQ_COLS['L']].notna().sum())}/{len(df)}")
    if missing:
        print(f"  !! {len(missing)} target(s) have NO fold YAML, so no sequences: "
              f"{sorted(missing)[:6]}" + (" ..." if len(missing) > 6 else ""))


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


def _add_leak_flags(targets_parquet, leak_parquet):
    """Join the pre-cutoff homology flags onto targets.parquet as booleans.

    The strict subset the card offers (130 targets) must be selectable from one table;
    the full evidence table (per-target identities and best-hit entries) ships alongside
    as leak_audit.parquet.
    """
    import pandas as pd
    if not leak_parquet.exists():
        print(f"  !! {leak_parquet} missing -- leak flags will not be in targets.parquet")
        return
    df = pd.read_parquet(targets_parquet)
    lk = pd.read_parquet(leak_parquet)
    for cutoff in ("pre2021", "pre2023"):
        m = dict(zip(lk.target, lk[f"flag_{cutoff}"] != ""))
        df[f"leak_flag_{cutoff}"] = df.pdb_id.map(m).fillna(False)
    df.to_parquet(targets_parquet, index=False)
    print(f"  leak flags joined: {int(df.leak_flag_pre2021.sum())} pre-2021, "
          f"{int(df.leak_flag_pre2023.sum())} pre-2023")


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
        if "dockq_dockq" in df.columns:
            vals["N_SCORABLE"] = str(df.loc[df.dockq_dockq.notna(), "target"].nunique())
    lk = out / "leak_audit.parquet"
    if lk.exists():
        import pandas as pd
        lf = pd.read_parquet(lk)
        vals["LEAK_FLAGGED_PRE2023"] = ", ".join(
            sorted(lf.loc[lf.flag_pre2023 != "", "target"]))
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

    for name in ("targets.parquet", "labels.parquet", "ensembles.parquet",
                 "leak_audit.parquet"):
        if not (out / name).exists():
            fail.append(f"{name} missing -- the dataset card documents all four tables")

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
        # The card promises every row traces to the exact code that produced it. Three values
        # break that promise, not one: a "-dirty" suffix (uncommitted edits on top of a real
        # sha), a null commit (a fold recovered from artifacts, whose driver knew the commit and
        # died with it), and "unknown" (git unreadable at fold time). Blocking only on "-dirty"
        # let the other two through, and they are strictly worse -- they name no sha at all.
        # Regenerate those folds rather than publish an untraceable row.
        commit = df.tt_bio_commit
        untraceable = df[commit.isna() | commit.astype(str).str.endswith("-dirty")
                         | commit.astype(str).isin(["unknown", "None", ""])]
        if len(untraceable):
            folds = sorted({f"{t}/{g}" for t, g in zip(untraceable.target, untraceable.generator)})
            fail.append(f"{len(untraceable)} rows ({len(folds)} folds) cannot be traced to a "
                        f"commit (dirty worktree, or recovered with the commit lost): {folds[:6]}"
                        + (" ..." if len(folds) > 6 else ""))

        # Does every released target actually have an interface a scorer can see? Written at
        # turn 77, wired here at turn 97: it had no caller for twenty passes, which is the same
        # way a positional join survived for weeks -- a check nobody runs is a comment.
        #
        # It belongs in the PREFLIGHT and not in the endgame's selftest step: 9ly2/9ly3/9lz2 have a
        # phosphoserine-mediated interface that DockQ cannot score, and whether to drop them is an
        # open decision. Blocking the whole chain on an undecided question would strand every other
        # step; blocking only the upload is exactly right, and means the question cannot be
        # forgotten at the moment it matters.
        #
        # Scoped to the targets actually present, so a deliberate subset release is not failed by a
        # target it does not contain.
        released = set(df.target.unique())
        audit_json = out / "_native_interface_audit.json"
        r = run([sys.executable, str(SCRIPTS / "abag_xm_native_interface_audit.py"),
                 "--json", str(audit_json)], capture_output=True, text=True)
        if not audit_json.exists():
            fail.append("native-interface audit did not run; cannot confirm the released targets "
                        f"have a scorable interface (rc={r.returncode})")
        else:
            # §1.1 report-with-exclusion: the anti-phosphoepitope targets stay IN the
            # release with null DockQ (their interface is carried by SEP residues the
            # DockQ loader discards -- abag-xm-label-census.md). Exempt exactly them,
            # and only while their released dockq really is all-null; any other
            # unscorable native fails the gate as before.
            KNOWN_UNSCORABLE = {"9ly2", "9ly3", "9lz2"}
            bad = [a for a in json.loads(audit_json.read_text())
                   if a.get("status") != "ok" and a.get("target") in released
                   and a.get("target") not in KNOWN_UNSCORABLE]
            if bad:
                named = ", ".join(f"{a['target']}({a['status']})" for a in bad[:6])
                fail.append(f"{len(bad)} released target(s) have no scorable interface: {named}"
                            + (" ..." if len(bad) > 6 else "")
                            + " -- their DockQ/epitope/lDDT labels have no referent")
            leaked = [t for t in KNOWN_UNSCORABLE & released
                      if df.loc[df.target == t, "dockq_dockq"].notna().any()]
            if leaked:
                fail.append(f"exempt unscorable target(s) {sorted(leaked)} carry non-null "
                            f"dockq -- the §1.1 exemption is stale; re-run the census")

        # Two of the tools this campaign uses may not be redistributed, and the release is
        # CC-BY-4.0: ABAG-Rank's weights and bundled examples are CC BY-NC 4.0, and PSBench's
        # models are AF3 outputs we do not own. Both were only ever meant to be internal scoring
        # tools. The licence gate has two halves: the banned columns must be ABSENT (asserted
        # here, not trusted to the data flow -- a breach is not the kind of thing to notice
        # after upload), and the one cleared learned ranker must be PRESENT -- DeepRank-Ab is
        # Apache-2.0 and Moritz's 2026-07-30 decision keeps it as the shipped learned-ranker
        # column, so an empty or missing deeprank_ab is as much a defect as a leak.
        banned = {c: t for c in df.columns
                  for t, pats in (("ABAG-Rank (weights CC BY-NC 4.0)", ("abag_rank", "abagrank")),
                                  ("PSBench (AF3 outputs, not ours)", ("psbench", "dockq_wave")))
                  if any(p in c.lower() for p in pats)}
        if banned:
            fail.append("labels.parquet carries non-redistributable columns -- "
                        + "; ".join(f"{c!r} from {t}" for c, t in sorted(banned.items())))
        if "deeprank_ab" not in df.columns:
            fail.append("deeprank_ab missing from labels.parquet -- the release ships no "
                        "learned-ranker column (DeepRank-Ab, Apache-2.0, is the cleared one)")
        elif df.deeprank_ab.isna().any():
            fail.append(f"deeprank_ab null in {int(df.deeprank_ab.isna().sum())}/{len(df)} "
                        f"rows -- the keyed join to ranker_scores.csv did not cover every sample")

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
        _add_fold_sequences(out / "targets.parquet")
    else:
        print(f"  !! {TARGETS_SRC} missing -- targets.parquet will not be in the release")
    leak_src = ROOT / "docs" / "abag-xm-leak-audit.parquet"
    if leak_src.exists() and (out / "targets.parquet").exists():
        shutil.copyfile(leak_src, out / "leak_audit.parquet")
        _add_leak_flags(out / "targets.parquet", out / "leak_audit.parquet")

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
