"""perf/kitcmp/bc2_traj.hook times every trajectory when several run on their own threads over one card."""
import sys, threading, time, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "perf" / "kitcmp"))
import bc2_traj  # noqa: E402


def test_interleaved_trajectories_each_get_their_own_record(tmp_path, monkeypatch):
    monkeypatch.setattr(bc2_traj, "loss_rows", lambda project: 0)
    rows = {"a": 7, "b": 3, "c": 5}

    def run_trajectory(*args, trajectory_directory, **kw):
        d = Path(trajectory_directory)
        d.mkdir()
        (d / f"{d.name}_losses.csv").write_text("i\n" + "".join(f"{i}\n" for i in range(rows[d.name])))
        time.sleep(0.05)

    campaign = types.SimpleNamespace(run_trajectory=run_trajectory,
                                     redesign_and_validate_binders=lambda *a, **k: time.sleep(0.02))
    emitted = []
    traj = bc2_traj.hook(campaign, tmp_path, emitted.append)

    def worker(names):
        for n in names:
            campaign.run_trajectory(trajectory_directory=str(tmp_path / n))
            campaign.redesign_and_validate_binders()

    ts = [threading.Thread(target=worker, args=(names,)) for names in (["a", "c"], ["b"])]
    [t.start() for t in ts]
    [t.join() for t in ts]
    traj.flush()
    traj.flush()
    assert sorted(emitted) == [0, 1, 2]
    by = {tr["design"]: tr for tr in traj}
    assert {k: v["iterations"] for k, v in by.items()} == rows
    assert all(tr["validate_s"] >= 0.02 and tr["design_s"] >= 0.05 for tr in traj)
