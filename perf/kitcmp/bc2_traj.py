"""Per-trajectory timing of BindCraft 2, shared by perf/spd/bench.py --model bindcraft2 (TT) and perf/kitcmp/bc2_gpu.py
(GPU) so the two sides cannot drift apart in what they call a trajectory's time or its iteration count.

`hook(campaign, project, emit)` wraps BindCraft 2's own `campaign.run_trajectory` and `redesign_and_validate_binders`
and returns the list of trajectories it has seen. Per trajectory: design_s = the `run_trajectory` call on a monotonic
clock, iterations = rows added to the project's losses.csv files during that call (BindCraft 2 writes one row per design
iteration and may zip a finished trajectory's folder, so every folder and archive is counted), loadavg = the 1-minute
load average at the start and end of the call. `emit(i)` fires for trajectory i once the next one starts; the caller
emits the last one itself. Seconds per iteration = design_s / iterations (`s_per_iter`).
"""
import contextlib, csv, os, time


def cpu_model():
    """The host CPU, which moves BindCraft 2's host JAX by ~20 % between the fleet's boxes."""
    with contextlib.suppress(OSError):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return None


def loss_rows(project):
    """Rows of every trajectory's losses.csv so far, live folders and zipped ones alike."""
    from bindcraft.campaign_output import trajectory_metric_rows
    with contextlib.suppress(Exception):
        return sum(len(r) for r in trajectory_metric_rows(str(project)))
    return 0


def s_per_iter(tr):
    return tr["design_s"] / tr["iterations"] if tr.get("iterations") and tr.get("design_s") else None


def table_row(project, n):
    """Trajectory n's row of BindCraft 2's trajectory table, and the campaign's accepted count so far."""
    from bindcraft.campaign_output import TRAJECTORY_STAGE, accepted_table, stage_table
    rows, acc = [], 0
    with contextlib.suppress(OSError):
        rows = list(csv.DictReader(open(stage_table(str(project), TRAJECTORY_STAGE))))
    with contextlib.suppress(OSError):
        acc = sum(1 for _ in csv.DictReader(open(accepted_table(str(project)))))
    return next((r for r in rows if r.get("trajectory") == str(n)), {}), acc


def hook(campaign, project, emit):
    traj = []
    real_design, real_validate = campaign.run_trajectory, campaign.redesign_and_validate_binders

    def design(*args, **kw):
        if traj:
            emit(len(traj) - 1)
        tr = dict(n=len(traj) + 1, t0=time.monotonic(), t0w=time.time(), rows0=loss_rows(project),
                  la0=os.getloadavg()[0])
        traj.append(tr)
        try:
            return real_design(*args, **kw)
        finally:
            tr["design_s"] = time.monotonic() - tr["t0"]
            tr["iterations"] = loss_rows(project) - tr["rows0"]
            tr["loadavg"] = [round(tr["la0"], 2), round(os.getloadavg()[0], 2)]

    def validate(*args, **kw):
        t0 = time.monotonic()
        try:
            return real_validate(*args, **kw)
        finally:
            traj[-1]["validate_s"] = time.monotonic() - t0

    campaign.run_trajectory, campaign.redesign_and_validate_binders = design, validate
    return traj
