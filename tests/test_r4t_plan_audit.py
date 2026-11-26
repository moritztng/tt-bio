"""The curve plan audit must refuse an exemption the plan's own data contradicts.

The audit itself is a gate, so it is shown able to fail here before any plan is trusted passing.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plan_audit", ROOT / "perf" / "r4t" / "plan_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

CHARGE = audit.PARAM_OF["charge"]
SS = audit.PARAM_OF["ss"]


def test_an_exemption_the_data_drives_is_reported():
    bad = audit.contradicted([CHARGE], {CHARGE: ["1bj6", "2plh"], SS: []})
    assert bad == {CHARGE: ["1bj6", "2plh"]}


def test_an_exemption_no_id_drives_is_not_reported():
    assert audit.contradicted([SS], {CHARGE: ["1bj6"], SS: []}) == {}


def test_a_driven_parameter_that_is_not_exempt_is_not_reported():
    # Driven and not exempt is the healthy case: the guard watches it.
    assert audit.contradicted([SS], {CHARGE: ["1bj6"], SS: []}) == {}


def test_matching_is_exact_not_by_prefix():
    # "init_repr_factory.feat_creators.1..." is a prefix of the charge table's name. A prefix or
    # substring test would read this exemption as covering feat_creators.16 and pass a real hole.
    near = "init_repr_factory.feat_creators.1.embedding.weight"
    assert audit.contradicted([near], {CHARGE: ["1bj6"]}) == {}
    assert audit.contradicted([CHARGE], {CHARGE: ["1bj6"]}) == {CHARGE: ["1bj6"]}


def test_the_graded_plan_on_disk_exempts_only_the_table_its_ids_cannot_drive():
    plan = json.loads((ROOT / "perf" / "r4t" / "plans" / "g3_curve.json").read_text())
    names = [e[0] for e in plan["exempt"]]
    assert names == [SS], names
    assert plan["steps"] >= 2000 and plan["crop"] == 256
    # The order is load-bearing: the guard only sees steps whose drawn depth is >= 2.
    assert plan["example_ids"][0].startswith("7rcu__4")


@pytest.mark.parametrize("drives,expect_fail", [({CHARGE: ["1bj6"]}, True), ({CHARGE: []}, False)])
def test_the_verdict_follows_the_measurement(drives, expect_fail):
    assert bool(audit.contradicted([CHARGE], drives)) is expect_fail
