#!/usr/bin/env python3
"""Controls for core_grid_sites.py."""
import ast
import json

import pytest

import core_grid_sites as C


def test_the_totals_partition_the_sites():
    t = C.analyse()["totals"]
    assert t["with_core_grid"] + t["no_core_grid_no_splat"] + t["no_core_grid_but_splat"] == t["sites"]
    assert t["sites"] >= 80


def test_ast_attributes_keywords_to_the_right_multiline_call(tmp_path, monkeypatch):
    """A grep on the call name cannot tell which keywords belong to which call when the calls span
    many lines and sit next to each other. Two adjacent calls, one with core_grid and one without."""
    p = tmp_path / "eng.py"
    p.write_text(
        "import ttnn\n"
        "def f(a, b, w, ckc):\n"
        "    x = ttnn.linear(\n        a, w,\n        compute_kernel_config=ckc,\n"
        "        core_grid=CORE_GRID_MAIN,\n    )\n"
        "    y = ttnn.linear(\n        b, w,\n        compute_kernel_config=ckc,\n    )\n"
        "    return x, y\n")
    monkeypatch.setattr(C, "ENGINE", p)
    sites = C.scan()
    assert len(sites) == 2
    assert [s["core_grid"] for s in sites] == [True, False]


def test_a_kw_splat_is_not_counted_as_bare(tmp_path, monkeypatch):
    p = tmp_path / "eng.py"
    p.write_text("import ttnn\ndef f(a, w, kw):\n    return ttnn.matmul(a, w, **kw)\n")
    monkeypatch.setattr(C, "ENGINE", p)
    r = C.analyse()
    assert r["totals"]["no_core_grid_but_splat"] == 1
    assert r["totals"]["no_core_grid_no_splat"] == 0


def test_a_source_where_everything_passes_core_grid_reports_no_bare_sites(tmp_path, monkeypatch):
    p = tmp_path / "eng.py"
    p.write_text("import ttnn\ndef f(a, w):\n    return ttnn.linear(a, w, core_grid=CORE_GRID_MAIN)\n")
    monkeypatch.setattr(C, "ENGINE", p)
    r = C.analyse()
    assert r["totals"]["no_core_grid_no_splat"] == 0
    assert r["bare_on_hot_path"] == []


# --- the finding -------------------------------------------------------------------------------
def test_the_hot_path_really_has_bare_sites_and_they_are_the_named_classes():
    r = C.analyse()
    hot = r["bare_on_hot_path"]
    assert len(hot) >= 10
    scopes = " ".join(s["scope"] for s in hot)
    for cls in ("TriangleMultiplication", "TriangleAttention", "OuterProductMean",
                "DiffusionTransformer", "Diffusion"):
        assert cls in scopes, f"{cls} missing from the hot-path list"


def test_the_fp32_path_is_counted_separately_not_folded_in():
    r = C.analyse()
    assert r["bare_on_separate_fp32_path"], "fp32 sites must be listed"
    for s in r["bare_on_separate_fp32_path"]:
        assert any(c.startswith("Fp32") for c in s["chain"])
    for s in r["bare_on_hot_path"]:
        assert not any(c.startswith("Fp32") for c in s["chain"])


def test_every_site_listed_is_really_bare():
    r = C.analyse()
    for key in ("bare_on_hot_path", "bare_on_separate_fp32_path", "bare_elsewhere"):
        for s in r[key]:
            assert s["core_grid"] is False and s["kw_splat"] is False


def test_the_line_numbers_point_at_a_real_matmul_call():
    src = C.ENGINE.read_text().splitlines()
    for s in C.analyse()["bare_on_hot_path"]:
        assert "ttnn.linear" in src[s["line"] - 1] or "ttnn.matmul" in src[s["line"] - 1]


# --- it must not become a priced lever ----------------------------------------------------------
def test_it_does_not_price_a_lever_and_says_why():
    r = C.analyse()
    joined = " ".join(r["what_this_does_NOT_establish"])
    assert "does not price a lever" in joined
    assert "one-size-tuning defect" in joined
    assert "PER SITE" in joined
    assert "4.908" not in r["answer"] or "NOT purely an artifact" in r["answer"]


def test_the_census_caveat_is_carried_verbatim_in_spirit():
    c = C.analyse()["census_lead"]
    assert c["difference_s"] == 4.908
    assert "not a fold saving" in c["caveat"]


def test_a_guard_that_fires_if_the_lever_is_ever_taken():
    """If the engine is changed so no hot-path site is bare, this note describes a fixed problem and
    must be rewritten rather than left standing."""
    assert C.analyse()["bare_on_hot_path"], "no bare hot-path sites left: rewrite this note"
