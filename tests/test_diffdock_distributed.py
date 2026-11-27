"""DiffDock through tt-bio's controller and leasing-worker path, graded without a card.

A docking screen is embarrassingly parallel across complexes, so `tt-bio dock` scales out the way
every other tt-bio model does: each complex becomes one controller job, and whichever workers are
connected dock them on the chips they already hold. `--devices 0,1,2` is the same path against a
private pool with one worker per card; `--controller URL` submits to a running fleet.

Everything here is real except the chip. The controller is the real `ControllerServer` on
loopback, the workers lease through the real `ControllerClient` and run the real worker shard
(`tt_bio.worker._execute_dock_job_inprocess`); a thread stands in for each card's process.
Docking itself is upstream's `--no_model` seed-conformer baseline (no weights, no device) or, where
the fan-out must load a network, a host stand-in for it. Whether the network docks well is graded
elsewhere (state/dfd/EVAL.md); this file grades that a screen reaches every card and comes back
whole, in the same layout as a local run.
"""
from __future__ import annotations

import contextlib
import csv
import threading
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from tt_bio import worker as worker_mod
from tt_bio.diffdock import fleet, run as dock_run
from tt_bio.diffdock.job import DockJob, DockOptions
from tt_bio.host_controller import ControllerClient, ControllerServer
from tt_bio.main import cli

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
LIGANDS = [ASPIRIN, "c1ccccc1O", "CCO", "CC(=O)Nc1ccc(O)cc1", "OC(=O)CCC(=O)O", "c1ccncc1"]


def _receptor(path: Path) -> Path:
    """A four-residue receptor: enough CA atoms for the seed baseline's centre."""
    xyz = [(10.0, 12.0, 8.0), (13.8, 12.0, 8.0), (13.8, 15.8, 8.0), (10.0, 15.8, 9.0)]
    lines = [f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
             f"           C" for i, (x, y, z) in enumerate(xyz)]
    path.write_text("\n".join(lines + ["END"]) + "\n")
    return path


@pytest.fixture
def screen(tmp_path):
    """A six-complex upstream CSV against one receptor, with one ligand shipped as a file."""
    rec = _receptor(tmp_path / "rec.pdb")
    smi = tmp_path / "paracetamol.smi"
    smi.write_text(LIGANDS[3] + "\n")
    rows = [{"complex_name": f"c{i}", "protein_path": str(rec), "protein_sequence": "",
             "ligand_description": str(smi) if i == 3 else lig} for i, lig in enumerate(LIGANDS)]
    path = tmp_path / "screen.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


@contextlib.contextmanager
def pool(cards, tmp_path, taken):
    """A loopback controller with one leasing worker per card; yields its URL.

    Each worker is the loop a card's process runs, minus the chip: lease a job, run the real
    dock shard, report back. ``taken`` collects (card, complex) as the jobs are handed out.
    """
    server = ControllerServer(0, tmp_path / f"controller-{len(cards)}.sqlite3")
    server.serve_in_background()
    url = f"http://127.0.0.1:{server.port}"
    stop = threading.Event()

    def serve(card):
        client = ControllerClient(url)
        info = {"worker_id": f"w{card}", "host": "test", "accelerator": "tenstorrent",
                "device_id": card, "label": f"card{card}", "model": None, "warm": None}
        while not stop.is_set():
            lease = client.lease(info, 1)
            if not lease.get("jobs"):
                time.sleep(0.02)
                continue
            for job in lease["jobs"]:
                taken.append((card, job["name"]))
                worker_mod._execute_dock_job_inprocess(client, lease["run_id"], info["worker_id"],
                                                       {"device": card}, job, lease["config"])

    threads = [threading.Thread(target=serve, args=(c,), daemon=True) for c in cards]
    for t in threads:
        t.start()
    deadline = time.time() + 10
    while int(ControllerClient(url).cluster().get("online_workers") or 0) < len(cards):
        assert time.time() < deadline, "the test workers never registered"
        time.sleep(0.02)
    try:
        yield url
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)
        server.shutdown()


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_job_through_controller(screen, tmp_path):
    """`tt-bio dock --controller URL` runs a screen on the fleet's workers and writes exactly
    what a local run writes: same complexes, same files, same bytes (the seed baseline with a
    fixed seed is deterministic), so a user cannot tell which path docked their screen."""
    flags = ["--no_model", "--diffusion_samples", "3", "--seed", "7"]
    local = CliRunner().invoke(cli, ["dock", str(screen), "--out_dir", str(tmp_path / "local"),
                                     *flags], catch_exceptions=False)
    assert local.exit_code == 0, local.output

    taken: list = []
    with pool([0], tmp_path, taken) as url:
        r = CliRunner().invoke(cli, ["dock", str(screen), "--out_dir", str(tmp_path / "fleet"),
                                     "--controller", url, "--owner", "alice", *flags],
                               catch_exceptions=False)
    assert r.exit_code == 0, r.output
    assert sorted(name for _, name in taken) == [f"c{i}" for i in range(6)]
    fleet_tree, local_tree = _tree(tmp_path / "fleet"), _tree(tmp_path / "local")
    assert set(fleet_tree) == set(local_tree)
    assert "c3/rank1.sdf" in fleet_tree, "a ligand given as a file must travel to the worker"
    assert fleet_tree == local_tree


def test_a_failed_complex_is_a_row_not_the_end_of_the_screen(tmp_path):
    """One bad ligand in a fleet screen lands in failures.csv and the rest still dock, as a
    local CSV run does; a run where nothing docked raises with the first reason."""
    rec = _receptor(tmp_path / "rec.pdb")
    jobs = [DockJob("good", receptor_path=rec, ligand=ASPIRIN),
            DockJob("bad", receptor_path=rec, ligand="not-a-smiles((")]
    options = DockOptions(no_model=True, diffusion_samples=2, seed=1)
    with pool([0], tmp_path, []) as url:
        summary = fleet.dock_via_controller(jobs, options, tmp_path / "out",
                                            controller_url=url, verbose=False)
        assert [d["name"] for d in summary["docked"]] == ["good"]
        assert [f[0] for f in summary["failed"]] == ["bad"]
        assert "bad" in (tmp_path / "out" / "failures.csv").read_text()
        assert (tmp_path / "out" / "good" / "rank1.sdf").is_file()
        with pytest.raises(RuntimeError, match="every complex failed; bad"):
            fleet.dock_via_controller(jobs[1:], options, tmp_path / "out2",
                                      controller_url=url, verbose=False)


def test_options_travel_to_the_worker_unchanged():
    """The run config carries the options as JSON; the worker rebuilds the same object."""
    import json

    o = DockOptions(diffusion_samples=40, pocket=(1.0, -2.5, 3.0), seed=11, sampler="ode",
                    checkpoint=Path("/w/score.pt"), recipe={**DockOptions().recipe, "x": 1})
    assert DockOptions.from_dict(json.loads(json.dumps(o.as_dict()))) == o


@pytest.fixture
def no_card(monkeypatch):
    """Four cards that are not there: the device probe answers as a 4-card host would."""
    def detect(device_ids, num_devices, max_workers):
        cards = [int(d) for d in device_ids.split(",")] if device_ids else [0, 1, 2, 3]
        return cards[:max_workers]

    monkeypatch.setattr("tt_bio.runtime.detect_tenstorrent_devices", detect)
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "0,1,2,3")    # restored after the test


@pytest.fixture
def host_network(monkeypatch):
    """The network a worker loads, stood in on the host: counts loads, docks the seed pose.

    The dock is slowed a little so one worker cannot drain the queue before the others poll,
    which is the saturated-queue case the fan-out exists for.
    """
    loads = []
    monkeypatch.setattr(worker_mod, "_DOCK_MODEL", (None, None))
    monkeypatch.setattr(dock_run, "load_model",
                        lambda options, **kw: loads.append(1) or object())
    from dataclasses import replace

    def sample(job, options, write_dir, receptor, model, *, cache=None):
        time.sleep(0.3)
        return dock_run._dock_one(job, replace(options, no_model=True), write_dir)

    monkeypatch.setattr(dock_run, "_sample_one", sample)
    return loads


def test_shards_across_devices(screen, tmp_path, monkeypatch, no_card, host_network):
    """`--devices 0,1,2` spreads the complexes over a private pool with one worker per card,
    every card takes work, each worker loads the network once rather than once per complex,
    and every complex comes back under out_dir in the local layout."""
    taken: list = []
    pools = []

    @contextlib.contextmanager
    def local_controller(cards, debug=False, **kw):
        pools.append(list(cards))
        with pool(cards, tmp_path, taken) as url:
            yield url

    monkeypatch.setattr("tt_bio.main._local_controller", local_controller)
    r = CliRunner().invoke(cli, ["dock", str(screen), "--out_dir", str(tmp_path / "out"),
                                 "--devices", "0,1,2", "--diffusion_samples", "2", "--seed", "3"],
                           catch_exceptions=False)
    assert r.exit_code == 0, r.output
    assert pools == [[0, 1, 2]]
    assert {card for card, _ in taken} == {0, 1, 2}, f"a card took no complex: {taken}"
    assert sorted(name for _, name in taken) == [f"c{i}" for i in range(6)]
    assert 1 <= len(host_network) <= 3, "the network is loaded per worker, not per complex"
    for i in range(6):
        assert (tmp_path / "out" / f"c{i}" / "rank1.sdf").is_file()


def test_one_card_or_one_complex_stays_in_process(screen, tmp_path, monkeypatch, no_card,
                                                  host_network):
    """No pool for one card: the process pins itself to that card and docks in-process, and a
    single complex never fans out however many cards are named."""
    import os

    monkeypatch.setattr("tt_bio.main._local_controller",
                        lambda *a, **k: pytest.fail("a one-card run started a pool"))
    r = CliRunner().invoke(cli, ["dock", str(screen), "--out_dir", str(tmp_path / "one"),
                                 "--devices", "2", "--diffusion_samples", "2"],
                           catch_exceptions=False)
    assert r.exit_code == 0, r.output
    assert os.environ["TT_VISIBLE_DEVICES"] == "2"
    assert len(host_network) == 1

    rec = Path(str(screen)).parent / "rec.pdb"
    r = CliRunner().invoke(cli, ["dock", str(rec), "--ligand", ASPIRIN, "--name", "solo",
                                 "--out_dir", str(tmp_path / "solo"), "--devices", "0,1,2",
                                 "--diffusion_samples", "2"], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    assert (tmp_path / "solo" / "solo" / "rank1.sdf").is_file()
