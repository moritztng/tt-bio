"""Docking fanned across cards and hosts through tt-bio's controller.

Docking is embarrassingly parallel across complexes, so the unit of work a card gets is one
complex: its receptor, its ligand and every pose of it. Poses of one complex stay on one card
because the sampler's per-step noise is one draw over the whole pose batch; splitting it would
be a different trajectory, not the same one run faster (state/dfd/SAMPLING.md).

    tt-bio dock ligands.csv --devices 0,1,2,3          # a private pool, one worker per card
    tt-bio dock ligands.csv --controller http://127.0.0.1:8765

Both run here. The receptor and the ligand ship inline with each job, so a worker needs no view
of the submitter's filesystem, and a complex lands under ``out_dir/<name>/`` exactly as a local
run writes it. A worker keeps the network resident across the complexes it is handed
(``tt_bio.worker._execute_dock_job_inprocess``).
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from tt_bio.diffdock.job import DockJob, DockOptions

__all__ = ["dock_via_controller", "job_payload", "job_from_payload"]


def _file(path) -> dict[str, str]:
    path = Path(path).expanduser()
    return {"filename": path.name, "text": path.read_text()}


def job_payload(job: DockJob) -> dict[str, Any]:
    """One complex as JSON: the receptor and the ligand carried by value, not by path."""
    ligand = job.ligand
    ligand_path = Path(str(ligand)).expanduser()
    return {
        "name": job.name,
        "receptor": _file(job.receptor_path) if job.receptor_path is not None else None,
        "receptor_sequence": job.receptor_sequence,
        "ligand": _file(ligand_path) if ligand_path.is_file() else None,
        "ligand_smiles": None if ligand_path.is_file() else ligand,
    }


def job_from_payload(payload: dict[str, Any], workdir: Path) -> DockJob:
    """The worker side of :func:`job_payload`: write the shipped files under ``workdir``."""
    def place(entry, sub):
        if not entry:
            return None
        name = Path(entry["filename"]).name          # never a path the submitter chose
        target = workdir / sub / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(entry["text"])
        return target

    ligand = place(payload.get("ligand"), "ligand")
    return DockJob(name=payload["name"],
                   receptor_path=place(payload.get("receptor"), "receptor"),
                   receptor_sequence=payload.get("receptor_sequence"),
                   ligand=str(ligand) if ligand is not None else payload.get("ligand_smiles"))


def dock_via_controller(jobs: list[DockJob], options: DockOptions, out_dir, *,
                        controller_url: str, owner: str | None = None,
                        run_id: str | None = None, verbose: bool = True) -> dict:
    """Fleet twin of :func:`tt_bio.diffdock.run.dock`, with the same summary and layout.

    Every complex is one job. A failed complex is a row in ``failures.csv`` and the screen goes
    on, as a local CSV run does; only a run in which nothing docked raises.
    """
    from tt_bio.diffdock import output
    from tt_bio.host_controller import connect_controller
    from tt_bio.main import _write_job_outputs

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    client, online = connect_controller(controller_url)
    payload_jobs = [{"id": f"dock_{i}_{job.name}", "name": job.name,
                     "input_b64": base64.b64encode(
                         json.dumps(job_payload(job)).encode()).decode()}
                    for i, job in enumerate(jobs)]
    config = {"kind": "dock", "engine": "diffdock", "options": options.as_dict()}
    run_payload = {"data": f"{len(jobs)} complex(es)", "out_dir": str(out_dir),
                   "result_dir": str(out_dir), "jobs": payload_jobs, "config": config,
                   "owner": owner}
    if run_id:
        run_payload["run_id"] = run_id
    run_id = client.create_run(run_payload)["run_id"]
    if verbose:
        print(f"Dispatched {len(jobs)} complex(es) to {online} worker(s) at {controller_url}",
              flush=True)

    done, failures, after = [], [], 0
    while True:
        snap = client.events(run_id, after)
        for ev in snap.get("events", []):
            after = max(after, int(ev.get("seq", after)))
            if ev.get("event") != "done":
                continue
            row = ev.get("row") or {}
            name = str(row.get("name") or row.get("id", ""))
            if row.get("status") == "ok":
                _write_job_outputs(client, run_id, str(row.get("id")), out_dir)
                done.append({"name": name, "files": list(row.get("files") or [])})
                if verbose:
                    print(f"[dock:{name}] done ({row.get('runtime_s', '?')}s)", flush=True)
            else:
                failures.append((name, (row.get("error") or "failed").strip()))
        status = snap.get("status")
        if status == "canceled":
            raise RuntimeError("the docking run was canceled on the controller")
        if status in ("ok", "failed"):
            break
        time.sleep(0.5)

    output.write_failures(out_dir, failures)
    if failures and not done:
        raise RuntimeError(f"every complex failed; {failures[0][0]}: {failures[0][1]}")
    return {"docked": done, "failed": failures, "out_dir": str(out_dir)}
