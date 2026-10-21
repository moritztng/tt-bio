"""Every column the grade reads must be one upstream can actually emit.

A column name that upstream never writes reads as `missing`, and a metric that is silently
missing is a metric nobody grades. This has already happened twice from names taken out of the
docs: `bond_length_metrics.frac_unreasonable` (upstream repeats the metric name:
`bond_length_metrics.bond_length.frac_unreasonable`) and `oracle.plddt` (there is no bare
`plddt`, only `overall_plddt` and `plddt_coi`). So the names are checked against the pinned
upstream checkout itself, not against prose.

Prefixes come from upstream's metric configs; oracle suffixes from the AF3 translation tables in
`rfproteina/metrics/oracles/oracle_metric.py`. Skipped if the reference clone is not on this box.
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perf" / "rfd4"))
import metrics as mspec  # noqa: E402

UPSTREAM = pathlib.Path("/home/moritz/rfd4-upstream/RFD4-Proteina")
PIN = pathlib.Path("/home/moritz/.coworker/state/rfd4/UPSTREAM.json")
yaml = pytest.importorskip("yaml")
pytestmark = pytest.mark.skipif(not UPSTREAM.exists(), reason="upstream reference clone not on this host")

# Suffixes a metric class emits that are not derivable from a config key.
RMSD_METRICS_EMIT = {"rmsd"}
# Written by the evaluate pipeline, not by any metric config.
PIPELINE_COLUMNS = {"pre_oracle_pass", "chain_of_interest", "foldseek_cluster_id"}
PASS_CRITERIA_EMIT = {"pass"}


def _config_keys() -> dict[str, str]:
    """metric config key -> which group defines it."""
    root = UPSTREAM / "configs" / "metrics"
    keys: dict[str, str] = {}
    for group in ("inline_metrics", "refolding_metrics", "refolding_oracles", "pass_criteria"):
        for path in (root / group).glob("*.yaml"):
            for key, val in (yaml.safe_load(path.read_text()) or {}).items():
                if isinstance(val, dict) and "_target_" in val:
                    keys[key] = group
    return keys


def _oracle_suffixes() -> set[str]:
    """Output keys from the AF3 -> flat-metrics tables, read out of the source, not retyped."""
    src = (UPSTREAM / "src/rfproteina/metrics/oracles/oracle_metric.py").read_text()
    tree = ast.parse(src)
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        name = getattr(node.targets[0], "id", None) or getattr(
            getattr(node.targets[0], "target", None), "id", None)
        if isinstance(node.targets[0], ast.AnnAssign):  # pragma: no cover - ast shape
            name = node.targets[0].target.id
        if not name or not name.startswith("_") or not name.endswith("METRICS"):
            continue
        for elt in getattr(node.value, "elts", []):
            if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2 and isinstance(elt.elts[1], ast.Constant):
                out.add(elt.elts[1].value)
    for node in ast.walk(tree):  # the annotated-assignment form of the same tables
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id.startswith("_") and node.target.id.endswith("METRICS"):
            for elt in getattr(node.value, "elts", []):
                if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2 and isinstance(elt.elts[1], ast.Constant):
                    out.add(elt.elts[1].value)
    assert "overall_plddt" in out and "min_ipae_coi" in out, "oracle tables not parsed"
    return out


def _all_columns() -> set[str]:
    cols = {c for c, _, _ in mspec.COMMON}
    for entries in mspec.SPECIFIC.values():
        cols |= {c for c, _, _ in entries}
    return cols


def test_the_upstream_clone_is_the_pinned_sha():
    head = (UPSTREAM / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        head = (UPSTREAM / ".git" / head.split()[1]).read_text().strip()
    assert head == json.loads(PIN.read_text())["sha"], "the reference clone has moved off the pin"


def test_every_graded_column_exists_upstream():
    keys, oracle_suffixes = _config_keys(), _oracle_suffixes()
    bad = []
    for col in sorted(_all_columns()):
        if col.startswith("geom.") or col in PIPELINE_COLUMNS:
            continue  # the harness's own columns, and the pipeline's
        prefix, _, suffix = col.partition(".")
        group = keys.get(prefix)
        if group is None:
            bad.append(f"{col}: no metric config named {prefix!r}")
        elif group == "pass_criteria" and suffix not in PASS_CRITERIA_EMIT:
            bad.append(f"{col}: a pass criterion emits {prefix}.pass, not {suffix!r}")
        elif group == "refolding_oracles" and not (
                suffix in oracle_suffixes or suffix.startswith("frac_under_")):
            bad.append(f"{col}: {suffix!r} is not an oracle output key")
        elif group == "refolding_metrics" and suffix not in RMSD_METRICS_EMIT:
            bad.append(f"{col}: a refolding comparison emits .rmsd, not {suffix!r}")
    assert not bad, "columns upstream never writes:\n  " + "\n  ".join(bad)


def test_the_observed_self_test_csv_has_every_oracle_free_column_we_grade():
    """The self-test's own evaluate CSV is the ground truth for the `design`-tag columns."""
    import csv
    csv_path = pathlib.Path("/home/moritz/.coworker/artifacts/rfd4-quality/selftest/eval0"
                            "/evaluation_metrics/per_example_metrics.csv")
    if not csv_path.exists():
        pytest.skip("self-test evaluate CSV not on this host")
    with csv_path.open() as fh:
        have = set(next(csv.reader(fh)))
    want = {c for c, _, _ in mspec.COMMON if not c.startswith("geom.")}
    assert want <= have, f"graded but absent from a real CSV: {sorted(want - have)}"


# Supplied by the refolding pipeline, not by the user's metrics_tags.
ROUTING = ("is_refold:", "exists_refold:", "oracle:")


def _gates() -> dict[str, dict]:
    """metric config key -> its tag gate, as upstream's Metric.matches_tags reads it."""
    root = UPSTREAM / "configs" / "metrics"
    gates: dict[str, dict] = {}
    for group in ("inline_metrics", "refolding_metrics", "refolding_oracles", "pass_criteria"):
        for path in (root / group).glob("*.yaml"):
            for key, val in (yaml.safe_load(path.read_text()) or {}).items():
                if isinstance(val, dict) and "_target_" in val:
                    gates[key] = {g: {t for t in (val.get(g) or []) if not t.startswith(ROUTING)}
                                  for g in ("required_tags_all", "required_tags_any", "prohibited_tags")}
    return gates


def _activates(gate: dict, tags: set[str]) -> bool:
    return (gate["required_tags_all"] <= tags
            and (not gate["required_tags_any"] or gate["required_tags_any"] & tags)
            and not (gate["prohibited_tags"] & tags))


def test_every_family_grades_only_columns_its_own_tags_activate():
    """A column upstream will not emit for this family's metrics_tags is a silent `missing`,
    and a silently missing metric is a metric nobody grades."""
    gates = _gates()
    bad = []
    for family, tags in mspec.FEATURE_TAGS.items():
        tagset = set(tags)
        for column, _, _ in mspec.columns_for(family):
            if column.startswith("geom.") or column in PIPELINE_COLUMNS:
                continue
            gate = gates.get(column.split(".")[0])
            if gate and not _activates(gate, tagset):
                bad.append(f"{family} ({', '.join(tags)}) grades {column}, which its tags do not activate: "
                           f"{ {k: sorted(v) for k, v in gate.items() if v} }")
    assert not bad, "columns a family's tags never produce:\n  " + "\n  ".join(bad)
