#!/usr/bin/env python3
"""The check D117 asks for: fail if OpenFold3's permutation alignment took its silent fallback.

Two halves, and neither is worth much without the other.

  LOG SCAN     every committed log in the tree is searched for upstream's fallback line. The
               line goes to `logging`'s handler of last resort, which is stderr, so any run
               whose stderr was captured carries it. A hit is a run that fell back.
  ARTIFACTS    this probe's own `runs/` directory is the one place where a recorded fallback is
               the INTENDED content -- three of its ten arms are controls that must fall back.
               Scanning it would make the check fail on its own evidence, so the scan skips it
               and this half asserts it instead, which is strictly the stronger test: every arm
               CONFIGURED to break the alignment must report a fallback, every other arm must
               report it completing, and the tag must agree with the configuration. A blanket
               exclusion would have been a hole.
  CANARY       the log scan alone cannot tell "no run fell back" from "the line never reaches
               a log". So this drives upstream's REAL `safe_multi_chain_permutation_alignment`
               into both of its catch tiers with a deliberately incomplete batch and asserts
               the guard reports it. Eight small tensors, no dataset, no checkpoint, under a
               second. If the canary stops firing, the scan's silence stops meaning anything.

Give `--batch <a real batch>` to add the positive half: a batch on which the alignment
COMPLETES, so the check is shown to separate the two outcomes and not merely to say "fallback"
whatever it is handed. Without it the run says so rather than implying the positive was tested.

Exit 0 only if the scan is clean and every assertion holds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from permalign_guard import AlignmentGuard  # noqa: E402

FALLBACK_LINE = "falling back to naive alignment"
LOG_SUFFIXES = (".log", ".out", ".txt", ".err")
# See ARTIFACTS above. Checked by `artifact_check`, not skipped.
SCAN_SKIP = "perf/of3t_permalign/runs/"


def log_scan(repo: Path) -> tuple[bool, list[str], int]:
    """Search every COMMITTED log for the fallback line. `git ls-files` so an uncommitted
    scratch file cannot make the check pass or fail by accident."""
    files = subprocess.run(["git", "-C", str(repo), "ls-files"],
                           capture_output=True, text=True, check=True).stdout.split("\n")
    logs = [f for f in files if f.endswith(LOG_SUFFIXES) and not f.startswith(SCAN_SKIP)]
    hits = []
    for f in logs:
        p = repo / f
        try:
            if FALLBACK_LINE in p.read_text(errors="ignore"):
                hits.append(f)
        except OSError:
            continue
    return (not hits), hits, len(logs)


def is_control(r: dict) -> bool:
    """Whether an arm was CONFIGURED to break the alignment, read from what it recorded doing.

    Deliberately not `"CONTROL" in tag`: a tag is prose, so that test passes or fails on a
    rename and silently mis-sorts a stale artifact written under an older name (it did -- an
    arm carrying `--drop-key` sat in `runs/` tagged `B_frozen_dropkey`, and the check called it
    a broken positive). The three ways an arm is made to fall back are each recorded in the
    arm's own `call`, so read those and let the tag be a label.
    """
    c = r["call"]
    return bool(c.get("drop_key") or c.get("collate") == "recurse"
                or c.get("forward_twice") or len(r.get("verdict_per_pass", [])) > 1)


def artifact_check(repo: Path) -> tuple[bool, list[str], int]:
    """Every arm in `runs/` reports the outcome its CONFIGURATION implies. See SCAN_SKIP."""
    d = repo / SCAN_SKIP
    bad, n = [], 0
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        n += 1
        want_fallback = is_control(r)
        got_fallback = r["verdict"]["fell_back_by_wrapper"]
        if want_fallback != got_fallback:
            bad.append(f"{r['tag']}: configured to "
                       f"{'fall back' if want_fallback else 'complete'}, "
                       f"got {'a fallback' if got_fallback else 'a completion'}")
        # The tag is not what decides the above, but a tag that disagrees with the
        # configuration is how a stale artifact announces itself. Both must hold.
        if want_fallback != ("CONTROL" in r["tag"]):
            bad.append(f"{r['tag']}: tag and configuration disagree on whether this is a "
                       f"control -- likely a stale artifact from an earlier naming")
        if not r["verdict"]["detectors_agree"]:
            bad.append(f"{r['tag']}: the log and wrapper detectors disagree")
    return (not bad), bad, n


def canary() -> dict:
    """Upstream's own wrapper, driven into both catch tiers, with the guard watching.

    The batch is deliberately incomplete -- it carries only what the last-resort branch itself
    reads -- so the full path raises on the first feature it wants and the naive path raises on
    `token_index`. The wrapper then reaches its last resort and ZEROES EVERY LOSS WEIGHT, which
    is the reason this matters: a run can lose its entire training signal here and emit two log
    lines. The exact key in each `KeyError` is not asserted, only that both tiers raised; a
    future upstream may read its features in another order.
    """
    from openfold3.core.utils import permutation_alignment as PA

    n_atom = 8
    batch = {
        "residue_index": torch.arange(4).view(1, 1, 4),
        "atom_mask": torch.ones(1, 1, n_atom),
        "loss_weights": {"mse": torch.tensor([[1.0]])},
        "ground_truth": {"atom_positions": torch.zeros(1, 1, n_atom, 3)},
    }
    pred = torch.zeros(1, 1, n_atom, 3)
    with AlignmentGuard(PA.__name__) as g:
        PA.safe_multi_chain_permutation_alignment(batch=batch, atom_positions_predicted=pred)
    v = g.verdict()
    v["losses_zeroed"] = all(float(w.abs().sum()) == 0.0
                             for w in batch["loss_weights"].values())
    return v


def positive(batch_path: Path) -> dict:
    """A batch the alignment completes on, so the check is shown to read both outcomes."""
    from openfold3.core.utils import permutation_alignment as PA
    from openfold3.core.utils.tensor_utils import tensor_tree_map

    batch = torch.load(batch_path, weights_only=False)
    perm = batch.pop("ref_space_uid_to_perm", None)
    batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
    if perm is not None:
        batch["ref_space_uid_to_perm"] = perm
    gt = batch["ground_truth"]["atom_positions"]
    pred = gt + torch.randn(gt.shape, generator=torch.Generator().manual_seed(1)) * 1.0
    with AlignmentGuard(PA.__name__) as g:
        PA.safe_multi_chain_permutation_alignment(batch=batch, atom_positions_predicted=pred)
    return g.verdict()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=HERE.parents[1])
    ap.add_argument("--batch", type=Path, help="a real batch, for the positive half")
    ap.add_argument("--skip-scan", action="store_true")
    a = ap.parse_args()
    fails = []

    if not a.skip_scan:
        clean, hits, n = log_scan(a.repo)
        print(f"log scan: {n} committed logs, {len(hits)} carrying the fallback line")
        for h in hits:
            print(f"  FELL BACK: {h}")
        if not clean:
            fails.append("a committed log shows the permutation alignment falling back")

        aok, abad, an = artifact_check(a.repo)
        print(f"artifacts: {an} arms in {SCAN_SKIP}, {len(abad)} disagreeing with their "
              f"configuration")
        for b in abad:
            print(f"  {b}")
        if not aok:
            fails.append("an arm in the probe's own runs/ disagrees with its configuration")

    v = canary()
    ok = (v["fell_back_by_wrapper"] and v["fell_back_by_log"] and v["detectors_agree"]
          and v["n_completed"] == 0 and len(v["errors"]) == 2 and v["losses_zeroed"])
    print(f"canary: fallback reported by wrapper {v['fell_back_by_wrapper']} and by log "
          f"{v['fell_back_by_log']}, {len(v['errors'])} errors, all losses zeroed "
          f"{v['losses_zeroed']}")
    for e in v["errors"]:
        print(f"  {e['error']}")
    if not ok:
        fails.append("the canary did not report a fallback upstream definitely took")

    if a.batch:
        p = positive(a.batch)
        print(f"positive ({a.batch.name}): completed {p['completed']}, "
              f"naive invoked {p['naive_fallback_invoked']}")
        if not p["completed"]:
            fails.append(f"the alignment fell back on {a.batch}, which it should complete on")
    else:
        print("positive: SKIPPED, no --batch given; this run did not show the check can read "
              "a completing alignment")

    for f in fails:
        print(f"FAIL: {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
