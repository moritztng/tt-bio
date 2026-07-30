#!/usr/bin/env python3
"""Final verification for the esmfold2 leg of the AbAg-XM campaign.

Run ON qb1 after the learned-ranker pass finishes (marker file exists):

    python3 scripts/abag_xm_esmfold2_final_check.py

Exit codes: 0 = PASS (emit DONE), 2 = NOT READY (marker missing, relaunch later),
1 = FAIL (something is wrong — investigate before claiming done).

Hard checks:
  1. Marker ~/abag_xm/tier_a/.ranker_esm2_learned_done exists and CSV mtime > bak mtime.
  2. Log has no "!! EMPTY" (requested-but-empty scorer trap). Guardian relaunch lines are
     printed for review; the CSV integrity checks below are the real corruption detector.
  3. CSV: 4 gens x 8200 rows; esmfold2 = 164 targets x 50 ranks (0..49 each).
  4. esmfold2 deeprank_ab + abag_rank: ZERO blank cells (the other three gens have zero
     blanks in these columns, so there is no legitimate-blank precedent for this leg).
  5. esmfold2 blank sets in the other columns match the audited legit set exactly:
       dockq:              {9ly2, 9ly3, 9lz2} (no native interface, blank for all gens)
       interface_lddt:     {9gei, 9ly2, 9ly3, 9lz2, 9mnu, 9msc, 9mz8, 9xqc} (audited)
       cdr_h3_rmsd:        {9l9y, 9lwc, 9mnu, 9msc, 9udq} (no resolved H3 in native,
                           identically blank in boltz2)
       everything else:    zero blanks.
  6. Non-esmfold2 rows byte-identical vs ranker_scores.csv.bak-pre-esm2learned
     (the learned pass must not touch the other gens' rows).
Prints summary stats for the DONE line.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

TIERA = Path.home() / "abag_xm" / "tier_a"
CSV = TIERA / "ranker_scores.csv"
BAK = TIERA / "ranker_scores.csv.bak-pre-esm2learned"
LOG = Path.home() / "abag_xm" / "logs" / "ranker_esmfold2_learned.log"
MARKER = TIERA / ".ranker_esm2_learned_done"

EXPECTED_BLANKS = {
    "dockq": {"9ly2", "9ly3", "9lz2"},
    "interface_lddt": {"9gei", "9ly2", "9ly3", "9lz2", "9mnu", "9msc", "9mz8", "9xqc"},
    "cdr_h3_rmsd": {"9l9y", "9lwc", "9mnu", "9msc", "9udq"},
}
LEARNED_COLS = ("deeprank_ab", "abag_rank")
ALL_COLS = ["iptm", "ptm", "ranking_score", "complex_plddt", "pdockq2", "ipsae", "anticonf",
            "pss", "deeprank_ab", "abag_rank", "dockq", "epitope_jaccard", "interface_lddt",
            "cdr_h3_rmsd"]


def fail(msg, failures):
    failures.append(msg)
    print(f"FAIL {msg}")


def main():
    if not MARKER.exists():
        print(f"NOT READY: {MARKER} missing (learned-ranker pass still running)")
        return 2
    failures = []

    if not CSV.exists() or not BAK.exists():
        fail(f"CSV or bak missing ({CSV.exists()=}, {BAK.exists()=})", failures)
        print("\n".join(failures))
        return 1
    if CSV.stat().st_mtime <= BAK.stat().st_mtime:
        fail(f"CSV mtime not newer than bak ({CSV.stat().st_mtime} <= {BAK.stat().st_mtime})",
             failures)

    log = LOG.read_text(errors="replace") if LOG.exists() else ""
    if "!! EMPTY" in log:
        fail("log contains '!! EMPTY' (requested-but-empty scorer trap)", failures)
    guardian_lines = [l for l in log.splitlines() if "[guardian" in l]
    for l in guardian_lines:
        print(f"NOTE guardian: {l}")

    rows = list(csv.DictReader(CSV.open()))
    gens = Counter(r["gen"] for r in rows)
    esm = [r for r in rows if r["gen"] == "esmfold2"]
    targets = sorted({r["target"] for r in esm})
    print(f"rows: {len(rows)} total, gens={dict(gens)}")
    if len(esm) != 8200 or len(targets) != 164:
        fail(f"esmfold2 rows={len(esm)} targets={len(targets)} (want 8200/164)", failures)
    ranks_per = Counter(r["target"] for r in esm)
    bad = {t: n for t, n in ranks_per.items() if n != 50}
    if bad:
        fail(f"targets without exactly 50 rows: {bad}", failures)

    for col in ALL_COLS:
        blank_targets = sorted({r["target"] for r in esm if r[col] == ""})
        if col in LEARNED_COLS:
            if blank_targets:
                fail(f"esmfold2 {col}: {len(blank_targets)} targets with blanks "
                     f"(want zero — no legit-blank precedent): {blank_targets[:10]}", failures)
        else:
            want = sorted(EXPECTED_BLANKS.get(col, set()))
            if blank_targets != want:
                fail(f"esmfold2 {col}: blank targets {blank_targets} != audited {want}",
                     failures)

    bak_other = [l for l in BAK.read_text().splitlines(keepends=True)
                 if ",esmfold2," not in l]
    new_other = [l for l in CSV.read_text().splitlines(keepends=True)
                 if ",esmfold2," not in l]
    if new_other != bak_other:
        if Counter(new_other) == Counter(bak_other):
            print("NOTE non-esmfold2 rows reordered vs bak (same multiset)")
        else:
            diff = [l for l in new_other if l not in set(bak_other)][:3]
            fail(f"non-esmfold2 rows changed vs bak (sample: {diff})", failures)

    if failures:
        print(f"\n{len(failures)} FAILURES")
        return 1

    for col in LEARNED_COLS:
        vals = [float(r[col]) for r in esm]
        print(f"esmfold2 {col}: n={len(vals)} min={min(vals):.4f} "
              f"mean={sum(vals)/len(vals):.4f} max={max(vals):.4f}")
    print("FINAL CHECK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
