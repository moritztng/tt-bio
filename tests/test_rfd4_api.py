"""`tt_bio.rfd4.design()`: the Python entry point, and that the CLI is the same call.

Upstream's equivalent is `rfd4.Proteina.from_pretrained(...)` plus `rfd4.generate(...)`. Here the
one call covers both, so what is checked is the orchestration: which inputs each design is built
from, what the engine is asked for, how the outputs are named, and that nothing diverges from
`tt-bio design --model rfd4`. The featurizer and the writer themselves are graded against
upstream by scripts/rfd4_port/, on inputs this test does not need.
"""
import pathlib
import types

import pytest

from tt_bio import rfd4
from tt_bio.rfd4 import io as rfd4_io
from tt_bio.rfd4 import run as rfd4_run


class StubEngine:
    """An engine that records what it was asked and returns nothing the writer reads."""

    spec = types.SimpleNamespace(nsteps=60)

    def __init__(self):
        self.calls = []

    def design(self, features, *, num_designs, nsteps, seed):
        self.calls.append({"tokens": features["tokens"], "num_designs": num_designs,
                           "nsteps": nsteps, "seed": seed})
        return f"output-{len(self.calls)}"


@pytest.fixture
def host(monkeypatch, tmp_path):
    """The host path stubbed out: spec -> two inputs, featurize, write_design(s)."""
    written = []

    def prepare_examples(spec, out_dir, *, only=None, validate=True):
        ids = [i for i in ("a", "b") if only is None or i in only]
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prepare_examples.validate = validate
        return [rfd4.DesignInput(i, out_dir / f"{i}.cif") for i in ids]

    def featurize(cif, *, seed, **kw):
        import torch

        return {"padding_mask": torch.ones(1, 7, dtype=torch.bool),
                "tokens": 7, "cif": pathlib.Path(cif).name, "seed": seed}, "atom_array"

    def write_design(output, features, atom_array, path, *, row=0):
        written.append((output, features["cif"], row, pathlib.Path(path)))
        return pathlib.Path(path)

    monkeypatch.setattr(rfd4_io, "prepare_examples", prepare_examples)
    monkeypatch.setattr(rfd4_io, "featurize", featurize)
    def write_designs(output, features, atom_array, paths):
        return [write_design(output, features, atom_array, p, row=k) for k, p in enumerate(paths)]

    monkeypatch.setattr(rfd4_io, "write_design", write_design)
    monkeypatch.setattr(rfd4_io, "write_designs", write_designs)
    monkeypatch.setattr(rfd4_io, "start_writer", lambda n: None)
    return types.SimpleNamespace(written=written, prepare_examples=prepare_examples,
                                 out=tmp_path / "designs", spec=tmp_path / "spec.yaml")


def test_design_is_exported():
    assert rfd4.design is rfd4_run.design


def test_one_call_writes_every_design_of_every_example(host):
    engine = StubEngine()
    paths = rfd4.design(host.spec, host.out, num_designs=2, seed=3, engine=engine)

    assert [p.name for p in paths] == ["a_0_generated.cif", "a_1_generated.cif",
                                       "b_0_generated.cif", "b_1_generated.cif"]
    assert all(p.parent == host.out for p in paths)
    assert [(c["num_designs"], c["seed"]) for c in engine.calls] == [(2, 3), (2, 3)]
    # Each example's own annotated CIF is what its designs were written from.
    assert [(cif, row) for _, cif, row, _ in host.written] == [("a.cif", 0), ("a.cif", 1),
                                                               ("b.cif", 0), ("b.cif", 1)]


def test_a_single_design_keeps_the_plain_name(host):
    paths = rfd4.design(host.spec, host.out, engine=StubEngine())
    assert [p.name for p in paths] == ["a_generated.cif", "b_generated.cif"]


def test_nsteps_defaults_to_the_recipe_and_an_override_is_passed_through(host):
    engine = StubEngine()
    rfd4.design(host.spec, host.out, engine=engine)
    rfd4.design(host.spec, host.out, nsteps=12, engine=engine)
    assert [c["nsteps"] for c in engine.calls] == [60, 60, 12, 12]


def test_only_selects_designs_and_validate_reaches_the_builder(host):
    engine = StubEngine()
    paths = rfd4.design(host.spec, host.out, only=["b"], validate=False, engine=engine)
    assert [p.name for p in paths] == ["b_generated.cif"]
    assert host.prepare_examples.validate is False


def test_an_annotated_cif_is_its_own_input_and_is_not_rebuilt(host, tmp_path):
    cif = tmp_path / "prepared.cif"
    cif.write_text("data_x\n")
    paths = rfd4.design(cif, host.out, annotated_cif=True, engine=StubEngine())
    assert [p.name for p in paths] == ["prepared_generated.cif"]
    assert host.written[0][1] == "prepared.cif"
    with pytest.raises(ValueError, match="annotated CIF is one design's input"):
        rfd4.design(cif, host.out, annotated_cif=True, only=["a"], engine=StubEngine())


def test_an_engine_passed_in_is_reused_and_no_device_is_opened(host, monkeypatch):
    """Sweeping seeds pays the weight load once; passing an engine must not load another."""
    monkeypatch.setattr(rfd4.engine, "get_engine",
                        lambda *a, **k: pytest.fail("design() opened a device of its own"))
    engine = StubEngine()
    for seed in range(3):
        rfd4.design(host.spec, host.out / f"seed{seed}", seed=seed, engine=engine)
    assert [c["seed"] for c in engine.calls] == [0, 0, 1, 1, 2, 2]


def test_the_cli_runs_the_same_call(host, monkeypatch):
    """`tt-bio design --model rfd4` must not grow a second copy of this loop."""
    from click.testing import CliRunner

    from tt_bio.main import cli

    seen = {}

    def fake_design(spec, out_dir="designs", **kw):
        seen.update(kw, spec=pathlib.Path(spec), out_dir=pathlib.Path(out_dir))
        return []

    monkeypatch.setattr(rfd4, "design", fake_design)
    host.spec.write_text("a: {}\n")
    result = CliRunner().invoke(cli, ["design", str(host.spec), "--model", "rfd4",
                                      "--out_dir", str(host.out), "--num_designs", "3",
                                      "--nsteps", "40", "--seed", "7", "--spec", "a,b",
                                      "--recipe", "high/midpeak", "--no_validate"])
    assert result.exit_code == 0, result.output
    assert seen["spec"] == host.spec and seen["out_dir"] == host.out
    assert (seen["num_designs"], seen["nsteps"], seen["seed"]) == (3, 40, 7)
    assert seen["only"] == ["a", "b"] and seen["validate"] is False
    assert seen["recipe"] == "high/midpeak"


def test_an_unknown_recipe_fails_before_anything_is_built(host):
    """The thinking budget is a user choice with a fixed vocabulary; a typo must not surface
    after a spec has been built and a device opened."""
    with pytest.raises(ValueError, match="unknown recycle strategy 'thinking_massive'"):
        rfd4.design(host.spec, host.out, recipe="thinking_massive")
    assert not host.out.exists()


def test_a_recipe_and_a_loaded_engine_are_mutually_exclusive(host):
    with pytest.raises(ValueError, match="already carries its recipe"):
        rfd4.design(host.spec, host.out, recipe="high/flat", engine=StubEngine())


def test_the_recipe_reaches_the_engine_that_is_loaded(host, monkeypatch):
    loaded = {}
    monkeypatch.setattr(rfd4.engine, "get_engine",
                        lambda **kw: loaded.update(kw) or StubEngine())
    rfd4.design(host.spec, host.out, recipe="medium/midpeak")
    assert loaded == {"recipe": "medium/midpeak"}


def test_bioir_is_declined_by_name_and_says_why(host):
    """Upstream's second backend is CUDA-only. A user who asks for it gets the reason, not a
    silent fall back to something else."""
    with pytest.raises(ValueError, match="NVIDIA BioNeMo-IR"):
        rfd4.design(host.spec, host.out, backend="bioir", engine=StubEngine())


def test_an_unknown_backend_names_the_ones_that_exist(host):
    with pytest.raises(ValueError, match=r"unknown inference backend 'cuda'.*bioir.*native"):
        rfd4.design(host.spec, host.out, backend="cuda", engine=StubEngine())


def test_the_cli_declines_bioir_without_opening_a_device(host, monkeypatch, tmp_path):
    from click.testing import CliRunner

    from tt_bio.main import cli

    monkeypatch.setattr(rfd4.engine, "get_engine",
                        lambda *a, **k: pytest.fail("a device was opened for a declined backend"))
    host.spec.write_text("a: {}\n")
    result = CliRunner().invoke(cli, ["design", str(host.spec), "--model", "rfd4",
                                      "--out_dir", str(host.out), "--backend", "bioir"])
    assert result.exit_code != 0
    assert "BioNeMo-IR" in result.output and "native" in result.output


def test_an_adapter_reaches_the_engine_and_a_missing_one_is_named(host, monkeypatch, tmp_path):
    loaded = {}
    monkeypatch.setattr(rfd4.engine, "get_engine",
                        lambda **kw: loaded.update(kw) or StubEngine())
    adapter = tmp_path / "ad"
    adapter.mkdir()
    rfd4.design(host.spec, host.out, adapter=adapter)
    assert loaded == {"adapter": adapter}
    with pytest.raises(ValueError, match="no LoRA adapter directory"):
        rfd4.design(host.spec, host.out, adapter=tmp_path / "nope")
    with pytest.raises(ValueError, match="already carries its recipe and adapter"):
        rfd4.design(host.spec, host.out, adapter=adapter, engine=StubEngine())


def test_lora_and_controller_together_are_refused_with_the_reason(host, tmp_path):
    from click.testing import CliRunner

    from tt_bio.main import cli

    adapter = tmp_path / "ad"
    adapter.mkdir()
    host.spec.write_text("a: {}\n")
    result = CliRunner().invoke(cli, ["design", str(host.spec), "--model", "rfd4",
                                      "--lora", str(adapter), "--controller", "http://c"])
    assert result.exit_code != 0
    assert "not shipped to fleet workers" in result.output
