"""One RFD4 design job fanned across the fleet: what is submitted, and what a worker runs.

Upstream scales out with SLURM array jobs and concatenates their per-job metrics CSVs. Here the
controller is the scale-out, so the two seams worth pinning without a card are the payload a
submitter builds (one job per design, each carrying its own annotated CIF and seed) and the shard
a worker executes from it. Both run against fakes; the designing itself is graded elsewhere.
"""
import base64
import json
import pathlib

import pytest

from tt_bio.rfd4 import fleet
from tt_bio.rfd4 import io as rfd4_io


class FakeClient:
    """A controller that accepts a run and reports every job done in one poll."""

    def __init__(self):
        self.payload = None

    def create_run(self, payload):
        self.payload = payload
        return {"run_id": "run-1"}

    def events(self, run_id, after):
        jobs = self.payload["jobs"]
        events = [{"seq": 1, "event": "start"}] + [
            {"seq": 2 + i, "event": "done",
             "row": {"id": j["id"], "name": j["name"], "status": "ok", "tokens": 128,
                     "runtime_s": 12.3, "example_id": j["name"].rsplit("_", 1)[0]}}
            for i, j in enumerate(jobs)]
        return {"events": events, "status": "ok"}

    def job_outputs(self, run_id, job_id):
        return {}


@pytest.fixture
def submitted(monkeypatch, tmp_path):
    """`design_via_controller` with the spec builder and the controller stubbed out."""
    client = FakeClient()
    monkeypatch.setattr("tt_bio.host_controller.connect_controller",
                        lambda url: (client, 3))

    def prepare_examples(spec, out_dir, *, only=None, validate=True):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        made = []
        for name in ("a", "b"):
            if only and name not in only:
                continue
            cif = out_dir / f"{name}.cif"
            cif.write_text(f"data_{name}\n")
            made.append(rfd4_io.DesignInput(name, cif))
        return made

    monkeypatch.setattr(rfd4_io, "prepare_examples", prepare_examples)
    spec = tmp_path / "spec.yaml"
    spec.write_text("a: {}\nb: {}\n")
    return client, spec, tmp_path / "out"


def payloads(client):
    return [json.loads(base64.b64decode(j["input_b64"]).decode()) for j in client.payload["jobs"]]


def test_every_design_is_its_own_job_carrying_its_own_input(submitted):
    client, spec, out = submitted
    rows = fleet.design_via_controller(spec, out, controller_url="http://c", num_designs=2,
                                       seed=5, verbose=False)

    assert [j["name"] for j in client.payload["jobs"]] == ["a_0", "a_1", "b_0", "b_1"]
    # Each job ships the annotated CIF the submitter built, so no worker reads this filesystem.
    assert [(p["example_id"], p["seed"], p["cif"]) for p in payloads(client)] == [
        ("a", 5, "data_a\n"), ("a", 6, "data_a\n"), ("b", 5, "data_b\n"), ("b", 6, "data_b\n")]
    assert client.payload["config"] == {"kind": "design", "engine": "rfd4"}
    assert [r["name"] for r in rows] == ["a_0", "a_1", "b_0", "b_1"]


def test_the_run_writes_the_metrics_every_design_landed_in(submitted):
    """Upstream concatenates per-job CSVs; one metrics.json is the same answer."""
    client, spec, out = submitted
    fleet.design_via_controller(spec, out, controller_url="http://c", verbose=False)
    metrics = json.loads((out / "metrics.json").read_text())
    assert [m["tokens"] for m in metrics] == [128, 128]
    assert [m["runtime_s"] for m in metrics] == [12.3, 12.3]


def test_the_recipe_rides_in_the_run_config_and_a_bad_one_never_reaches_the_fleet(submitted):
    client, spec, out = submitted
    fleet.design_via_controller(spec, out, controller_url="http://c", recipe="high/flat",
                                nsteps=40, verbose=False)
    assert client.payload["config"]["recipe"] == "high/flat"
    assert client.payload["config"]["nsteps"] == 40
    assert "fast" not in client.payload["config"]
    fleet.design_via_controller(spec, out, controller_url="http://c", fast=True, verbose=False)
    assert client.payload["config"]["fast"] is True

    client.payload = None
    with pytest.raises(ValueError, match="unknown recycle strategy"):
        fleet.design_via_controller(spec, out, controller_url="http://c", recipe="nope")
    with pytest.raises(ValueError, match="NVIDIA BioNeMo-IR"):
        fleet.design_via_controller(spec, out, controller_url="http://c", backend="bioir")
    assert client.payload is None


def test_only_selects_which_designs_are_dispatched(submitted):
    client, spec, out = submitted
    fleet.design_via_controller(spec, out, controller_url="http://c", only=["b"], verbose=False)
    assert [j["name"] for j in client.payload["jobs"]] == ["b"]


def test_a_run_where_every_design_failed_raises_with_the_first_error(submitted, monkeypatch):
    client, spec, out = submitted
    monkeypatch.setattr(client, "events", lambda run_id, after: {
        "status": "failed",
        "events": [{"seq": 1, "event": "done",
                    "row": {"id": "rfd4_a", "name": "a", "status": "failed",
                            "error": "selection matched no atoms"}}]})
    with pytest.raises(RuntimeError, match="selection matched no atoms"):
        fleet.design_via_controller(spec, out, controller_url="http://c", verbose=False)


# --- the worker side -------------------------------------------------------------------

class WorkerClient:
    """Collects what a shard reports back."""

    def __init__(self):
        self.events, self.row, self.outputs = [], None, None

    def event(self, run_id, worker_id, body):
        self.events.append(body)

    def complete(self, run_id, worker_id, row, body, outputs=None):
        self.row, self.outputs = row, outputs


def test_a_worker_designs_from_the_shipped_cif_and_reports_tokens(monkeypatch, tmp_path):
    """The shard must design from the CIF in its payload and keep the engine across shards."""
    from tt_bio import rfd4
    from tt_bio import worker as worker_mod

    seen = {}

    def fake_design(spec, out_dir, *, num_designs, nsteps, seed, annotated_cif, engine,
                    on_design=None, **kw):
        seen.update(cif=pathlib.Path(spec).read_text(), seed=seed, nsteps=nsteps,
                    annotated_cif=annotated_cif, engine=engine)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(out_dir) / "a_generated.cif").write_text("data_out\n")
        if on_design:
            on_design("a", 128, pathlib.Path(spec))
        return []

    engines = []
    monkeypatch.setattr(rfd4, "design", fake_design)
    monkeypatch.setattr(worker_mod, "_rfd4_engine",
                        lambda cfg: engines.append(cfg.get("recipe")) or "ENGINE")

    client = WorkerClient()
    payload = {"stem": "a", "seed": 7, "cif": "data_in\n", "example_id": "a"}
    job = {"id": "rfd4_a", "name": "a",
           "input_b64": base64.b64encode(json.dumps(payload).encode()).decode()}
    worker_mod._execute_rfd4_job_inprocess(client, "run-1", "w1", {}, {}, job,
                                           {"engine": "rfd4", "recipe": "high/flat"})

    assert client.row["status"] == "ok", client.row.get("error")
    assert (seen["cif"], seen["seed"], seen["annotated_cif"]) == ("data_in\n", 7, True)
    assert seen["engine"] == "ENGINE" and engines == ["high/flat"]
    assert client.row["tokens"] == 128 and client.row["example_id"] == "a"
    assert "a_generated.cif" in (client.outputs or {})


def test_a_shard_with_no_input_fails_with_a_reason_not_a_traceback(monkeypatch):
    from tt_bio import worker as worker_mod

    client = WorkerClient()
    job = {"id": "rfd4_a", "name": "a",
           "input_b64": base64.b64encode(json.dumps({"stem": "a", "seed": 0}).encode()).decode()}
    worker_mod._execute_rfd4_job_inprocess(client, "run-1", "w1", {}, {}, job, {"engine": "rfd4"})
    assert client.row["status"] == "failed"
    assert "no annotated CIF" in client.row["error"]


def test_a_held_engine_is_dropped_before_a_predict_model_loads(monkeypatch):
    """The worker keeps the design engine across shards, so something has to let go of it
    when the same chip is asked for a fold."""
    from tt_bio import worker as worker_mod

    monkeypatch.setattr(worker_mod, "_RFD4_ENGINE", ("high/flat", object()))
    worker_mod._free_rfd4_engine()
    assert worker_mod._RFD4_ENGINE == (None, None)


def test_fast_keys_the_held_engine_and_is_set_before_it_builds(monkeypatch):
    """`--fast` picks the trunk's lever set when the engine is built, so a run that changes it
    gets a new engine built under it, and a run that keeps it reuses the held one."""
    from tt_bio import rfd4, tenstorrent
    from tt_bio import worker as worker_mod

    built = []
    monkeypatch.setattr(worker_mod, "_RFD4_ENGINE", (None, None))
    monkeypatch.setattr(rfd4, "get_engine", lambda **kw: built.append(tenstorrent._FAST_MODE) or object())
    monkeypatch.setattr(tenstorrent, "_FAST_MODE", False)
    a = worker_mod._rfd4_engine({"engine": "rfd4"})
    assert worker_mod._rfd4_engine({"engine": "rfd4"}) is a
    b = worker_mod._rfd4_engine({"engine": "rfd4", "fast": True})
    assert b is not a and built == [False, True]


def test_the_cli_submits_rather_than_designing_locally(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from tt_bio.main import cli
    from tt_bio.rfd4 import fleet as fleet_mod

    seen = {}
    monkeypatch.setattr(fleet_mod, "design_via_controller",
                        lambda spec, out_dir, **kw: seen.update(kw, spec=pathlib.Path(spec)) or [])
    monkeypatch.setattr("tt_bio.rfd4.run.design",
                        lambda *a, **k: pytest.fail("the CLI designed locally with --controller"))
    spec = tmp_path / "spec.yaml"
    spec.write_text("a: {}\n")
    result = CliRunner().invoke(cli, ["design", str(spec), "--model", "rfd4",
                                      "--out_dir", str(tmp_path / "out"), "--num_designs", "4",
                                      "--seed", "3", "--recipe", "high/flat",
                                      "--controller", "http://c", "--run-id", "job-9",
                                      "--owner", "abc"])
    assert result.exit_code == 0, result.output
    assert seen["controller_url"] == "http://c" and seen["num_designs"] == 4
    assert (seen["seed"], seen["recipe"], seen["run_id"], seen["owner"]) == (3, "high/flat",
                                                                            "job-9", "abc")


@pytest.mark.parametrize("fast", [True, False])
def test_the_cli_fast_flag_reaches_rfd4(monkeypatch, tmp_path, fast):
    """`--fast` is RFD4's lever set, not an ignored flag: set before the engine builds locally, and
    carried in a fleet run's config."""
    from click.testing import CliRunner

    from tt_bio import tenstorrent
    from tt_bio.main import cli
    from tt_bio.rfd4 import fleet as fleet_mod

    seen = {}
    monkeypatch.setattr(tenstorrent, "_FAST_MODE", False)
    monkeypatch.setattr("tt_bio.rfd4.design",
                        lambda *a, **k: seen.update(local=tenstorrent._FAST_MODE) or [])
    monkeypatch.setattr(fleet_mod, "design_via_controller",
                        lambda spec, out_dir, **kw: seen.update(fleet=kw["fast"]) or [])
    spec = tmp_path / "spec.yaml"
    spec.write_text("a: {}\n")
    args = ["design", str(spec), "--model", "rfd4", "--out_dir", str(tmp_path / "out")] + (["--fast"] if fast else [])
    for extra in ([], ["--controller", "http://c"]):
        result = CliRunner().invoke(cli, args + extra)
        assert result.exit_code == 0, result.output
        assert "does not use --fast" not in result.output
    assert seen == {"local": fast, "fleet": fast}
