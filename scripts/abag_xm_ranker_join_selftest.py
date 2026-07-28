#!/usr/bin/env python3
"""Assert the ranker table joins per-sample scores to per-sample labels BY RANK, not by position.

This guards the bug that made 48 of every 50 rows carry another structure's DockQ. The two sides of
the join are ordered differently and always were: `all_runs`, `pairwise_matrix` and `deeprank_batch`
are rank-ordered, while `labels.py::_samples()` sorts the model files by filename, so its list runs
0, 1, 10, 11, ..., 2, 20, ... Pairing them positionally agrees only at ranks 0 and 1.

The reason it needs a permanent test rather than a comment: the damage is invisible in aggregate.
A within-target permutation leaves between-target signal untouched, so global Spearman stayed 0.79
while the per-target median -- the quantity the dataset exists to report -- collapsed to 0.06, which
reads as a finding about confidence rather than as a bug. Nothing downstream can tell the two apart.

The fixture is built so position and rank DISAGREE (12 samples, so the filename order is
0, 1, 10, 11, 2, 3, ...) and each rank carries a distinct value on both sides. A positional join
therefore has to produce visibly wrong pairs, and the test asserts that too -- a check only ever
exercised on its passing case can be one that cannot fail.

    python3 scripts/abag_xm_ranker_join_selftest.py      # exit 0 = join is rank-keyed
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N = 12


def _load_ranker_scores():
    spec = importlib.util.spec_from_file_location(
        "abag_xm_ranker_scores", ROOT / "scripts" / "abag_xm_ranker_scores.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _lexicographic_ranks(n):
    """The order labels.py::_samples() produces: rank 0 first, then model files by NAME."""
    return [0] + sorted(range(1, n), key=str)


def _fixture(tmp):
    """A fold whose label list is in filename order and whose confidences are in rank order."""
    fold = tmp / "fold"
    (fold / "structures").mkdir(parents=True)

    # rank r: dockq = r/100, iptm = 1 - r/100. Distinct per rank and anti-correlated, so a
    # mispairing cannot coincidentally look right.
    dockq = {r: round(r / 100, 4) for r in range(N)}
    iptm = {r: round(1 - r / 100, 4) for r in range(N)}

    (fold / "results.json").write_text(json.dumps([{
        "id": "TEST", "status": "ok", "samples": N,
        "all_runs": [{"rank": r, "iptm": iptm[r], "ptm": 0.5,
                      "confidence_score": 0.5, "complex_plddt": 0.5}
                     for r in range(N)],   # rank-ordered, as the generator writes it
    }]))

    order = _lexicographic_ranks(N)
    assert order != list(range(N)), "fixture must exercise a disagreeing order"
    labels = {
        "target": "TEST", "n_samples": N,
        # samples in FILENAME order, each carrying its own rank -- exactly what labels.py writes
        "samples": [{"rank": r,
                     "cif": f"TEST.cif" if r == 0 else f"TEST_model_{r}.cif",
                     "dockq": {"dockq": dockq[r]},
                     "epitope_jaccard": {"epitope_jaccard": dockq[r]},
                     "interface_lddt": {"interface_lddt": dockq[r]},
                     "cdr_rmsd": {"cdrs": {"H3": dockq[r]}},
                     "pae_metrics": {"pdockq2": dockq[r], "ipsae": dockq[r],
                                     "anticonf": dockq[r]}}
                    for r in order],
        # pairwise matrix indices are ranks (it builds cifs with `for k in range(1, n)`)
        "pairwise_matrix": {"n_samples": N,
                            "matrix": [{"i": i, "j": j, "dockq": 0.5}
                                       for i in range(N) for j in range(i + 1, N)]},
        "basin_clust": {},
    }
    labels_path = tmp / "labels.json"
    labels_path.write_text(json.dumps(labels))
    return fold, labels_path, dockq, iptm, order


def main():
    m = _load_ranker_scores()
    fails = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fold, labels_path, dockq, iptm, order = _fixture(tmp)
        print(f"fixture: {N} samples; label list order (by filename) = {order}")

        rows = m._score_one_fold(fold, "TEST", "test-gen", labels_path,
                                 False, False, None, None)
        if len(rows) != N:
            fails.append(f"expected {N} rows, got {len(rows)}")

        wrong_label, wrong_score = [], []
        for row in rows:
            r = row["rank"]
            if abs(float(row["dockq"]) - dockq[r]) > 1e-9:
                wrong_label.append((r, row["dockq"], dockq[r]))
            if abs(float(row["iptm"]) - iptm[r]) > 1e-9:
                wrong_score.append((r, row["iptm"], iptm[r]))
        if wrong_label:
            fails.append(f"{len(wrong_label)} rows carry another rank's LABEL, "
                         f"e.g. rank {wrong_label[0][0]}: got {wrong_label[0][1]}, "
                         f"expected {wrong_label[0][2]}")
        if wrong_score:
            fails.append(f"{len(wrong_score)} rows carry another rank's SCORE, "
                         f"e.g. rank {wrong_score[0][0]}: got {wrong_score[0][1]}, "
                         f"expected {wrong_score[0][2]}")
        print(f"  rank-keyed join: {N - len(wrong_label)}/{N} labels and "
              f"{N - len(wrong_score)}/{N} scores land on the right rank")

        # The negative control. Reproduce what a positional join would have produced and require
        # this fixture to expose it -- otherwise a passing test proves nothing.
        positional_bad = sum(1 for pos, r in enumerate(order) if dockq[r] != dockq[pos])
        print(f"  negative control: a positional join mispairs {positional_bad}/{N} rows "
              f"on this fixture")
        if positional_bad == 0:
            fails.append("fixture cannot distinguish a positional join from a rank join; "
                         "the test would pass either way and is worthless")

    if fails:
        print("\nFAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS: per-sample scores and labels are joined by rank")
    return 0


if __name__ == "__main__":
    sys.exit(main())
