"""`aggregated_metrics.csv`: upstream's per-dataset reductions over the per-example rows.

Expected values are computed by hand from upstream's rules (rfproteina `proteina.py:1436-1503`,
`OracleMetric.aggregate`, `compute_solve_rate_metrics`): a median where the metric's config says so
and a mean otherwise, `frac_under_<T>` on `min_ipae_coi` with a strict `<`, NaN and empty cells
dropped before `n` is counted, and solve rates over per-target pass means.
"""
import math

import pytest

from tt_bio.rfd4.evaluate import aggregate


def _rows():
    ipae = [0.5, 1.0, 1.7, float("nan"), 3.0]  # 1.0 is not under 1: the threshold is strict
    rows = []
    for i, x in enumerate(ipae):
        rows.append({"example_id": f"d{i}", "dataset_name": "ds", "task": "t1" if i < 3 else "t2",
                     "designed_structure_path": f"/x/d{i}.cif", "metrics_tags": "ppi_refolding",
                     "oracle_ppi.min_ipae_coi": x, "oracle_ppi.iptm": 0.1 * (i + 1),
                     "binder_helix_fraction": 0.2 * i, "ppi.pass": 1.0 if i in (0, 3) else 0.0})
    return rows


def _table(rows):
    columns = list(rows[0])
    return {(a["metric"], a["reduction"]): (a["value"], a["n"]) for a in aggregate(rows, columns)}


def test_oracle_columns_take_a_median_and_frac_under_rows():
    t = _table(_rows())
    assert t[("oracle_ppi.min_ipae_coi", "median")] == (1.35, 4)  # NaN dropped: median of 0.5 1.0 1.7 3.0
    assert t[("oracle_ppi.min_ipae_coi", "frac_under_1")] == (0.25, 4)
    assert t[("oracle_ppi.min_ipae_coi", "frac_under_1.5")] == (0.5, 4)
    assert t[("oracle_ppi.min_ipae_coi", "frac_under_2")] == (0.75, 4)
    assert t[("oracle_ppi.iptm", "median")] == (0.3, 5)
    assert not any(m == "oracle_ppi.iptm" and r.startswith("frac_under") for m, r in t)


def test_other_columns_take_a_mean_and_identity_columns_none():
    t = _table(_rows())
    assert t[("binder_helix_fraction", "mean")] == (0.4, 5)
    assert not any(m in ("example_id", "task", "metrics_tags", "designed_structure_path") for m, _ in t)


def test_solve_rates_are_over_targets_not_designs():
    t = _table(_rows())
    # per-target pass means: t1 = 1/3, t2 = 1/2
    p = "ppi.solve_rate_"
    assert t[(p + "n_targets", "")] == (2.0, 2)
    assert t[(p + "any", "")][0] == 1.0
    assert t[(p + "20pct", "")][0] == 1.0
    assert t[(p + "auc", "")][0] == 1.0  # both capped at 0.20
    rows = _rows()
    for r in rows[:3]:
        r["ppi.pass"] = 0.0
    t = _table(rows)
    assert t[(p + "any", "")][0] == 0.5
    assert t[(p + "auc", "")][0] == pytest.approx(0.5)


def test_a_column_with_no_value_writes_no_row():
    rows = _rows()
    for r in rows:
        r["oracle_ppi.iptm"] = None
    assert not any(m == "oracle_ppi.iptm" for m, _ in _table(rows))
    assert math.isfinite(_table(rows)[("binder_helix_fraction", "mean")][0])
