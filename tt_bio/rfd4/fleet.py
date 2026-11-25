"""One design job fanned across the fleet.

Upstream scales out with SLURM: ``launch_generate.sh --njobs N`` writes one
``per_example_metrics_job{N}.csv`` per array task and ``aggregate_metrics`` concatenates them
afterwards. tt-bio has a controller and a job queue already, so the same scale-out is
``--controller``: every design in the spec, times ``--num_designs``, becomes one job, and whichever
workers are connected run them on the chips they already hold.

    tt-bio design specs.yaml --model rfd4 --num_designs 32 --controller http://127.0.0.1:8765

The inputs are built HERE, once, and each annotated CIF ships inline with its job. A worker
therefore needs no view of the submitter's filesystem and no copy of the spec's input structures,
and a spec that selects nothing fails once on the submitter instead of N times on N workers. It
also pins what ran: the CIF a worker designed from is the CIF this host wrote and kept.

Each design gets seed ``seed + k``, the same per-design seeding the local path uses, so a sweep is
reproducible and a single design out of a 32-design run can be reproduced on its own.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

__all__ = ["design_via_controller"]


def _jobs(inputs, num_designs: int, seed: int, workers: int) -> list[dict[str, Any]]:
    """One job per worker per design id, each carrying a BATCH of designs and its input CIF.

    ``num_designs`` is RFD4's batch axis: one trajectory of 32 designs costs far less than 32
    trajectories of one, because the trunk passes are shared across the batch. Shipping one
    design per job threw that away -- on a 4-card box, 32 designs ran 57 s batched on one card
    and 108 s as 32 single-design jobs across four (2026-11-25). So the unit of work a card
    gets is a slice of the run, not a design, and the slice carries the index of its first
    design: ``first .. first + count - 1`` are named by their index in the whole run and take
    that index's per-design noise, so a slice is the same slice whatever card runs it. Designs
    are samples, so a run split four ways is not the same set a single batch of 32 produces.
    """
    jobs = []
    per_job = max(1, -(-int(num_designs) // max(1, int(workers))))
    for design_input in inputs:
        cif = design_input.cif.read_text()
        for first in range(0, num_designs, per_job):
            count = min(per_job, num_designs - first)
            stem = f"{design_input.example_id}" + ("" if num_designs == 1 else f"_{first}")
            payload = {"stem": design_input.example_id, "seed": int(seed), "cif": cif,
                       "example_id": design_input.example_id,
                       "first": first, "count": count, "total": int(num_designs)}
            jobs.append({"id": f"rfd4_{stem}", "name": stem,
                         "input_b64": base64.b64encode(
                             json.dumps(payload).encode()).decode()})
    return jobs


def design_via_controller(spec: str | Path, out_dir: str | Path, *, controller_url: str,
                          num_designs: int = 1, backend: str = "native",
                          recipe: str | None = None,
                          nsteps: int | None = None, seed: int = 0,
                          only: list[str] | None = None, validate: bool = True,
                          annotated_cif: bool = False, fast: bool = False, run_id: str | None = None,
                          owner: str | None = None, shards: int | None = None,
                          verbose: bool = True) -> list[dict]:
    """Fleet twin of :func:`tt_bio.rfd4.design`. Returns one row per design written.

    ``shards`` is how many slices the designs are split into, one per card the run should use;
    it defaults to the workers online right now. A local pool passes its card count, because a
    worker that is still opening its chip is not online yet and would otherwise get no designs.

    A row is ``{"cif", "name", "example_id", "tokens", "runtime_s"}``. The CIFs land under
    ``out_dir`` exactly as the local path names them, and ``out_dir/metrics.json`` collects the
    rows, which is this fleet's answer to upstream's concatenated per-job metrics CSVs.
    """
    from tt_bio.host_controller import connect_controller
    from tt_bio.main import _write_job_outputs
    from tt_bio.rfd4.io import resolve_inputs
    from tt_bio.rfd4.run import check_backend
    from tt_bio.rfd4.sampler import recycle_strategy

    check_backend(backend)
    if recipe is not None:
        recycle_strategy(recipe)  # refuse an unknown budget before the fleet is touched

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = resolve_inputs(spec, out_dir / "inputs", only=only, validate=validate,
                            annotated_cif=annotated_cif)

    client, online = connect_controller(controller_url)
    config: dict[str, Any] = {"kind": "design", "engine": "rfd4"}
    if recipe is not None:
        config["recipe"] = recipe
    if nsteps is not None:
        config["nsteps"] = int(nsteps)
    if fast:
        config["fast"] = True

    jobs = _jobs(inputs, int(num_designs), int(seed), int(shards or online or 1))
    run_payload = {"data": "rfd4-design", "out_dir": str(out_dir), "result_dir": str(out_dir),
                   "jobs": jobs, "config": config, "owner": owner}
    if run_id:
        run_payload["run_id"] = run_id
    run_id = client.create_run(run_payload)["run_id"]
    if verbose:
        print(f"Dispatched {len(jobs)} design(s) across {online} worker(s) on the fleet at "
              f"{controller_url}", flush=True)

    rows: list[dict] = []
    failures: dict[str, str] = {}
    after, announced = 0, False
    while True:
        snap = client.events(run_id, after)
        for ev in snap.get("events", []):
            after = max(after, int(ev.get("seq", after)))
            if ev.get("event") == "start" and not announced:
                announced = True
                print("stage: design", flush=True)
            elif ev.get("event") == "done":
                row = ev.get("row") or {}
                name = str(row.get("name") or row.get("id", ""))
                if row.get("status") == "ok":
                    _write_job_outputs(client, run_id, str(row.get("id")), out_dir)
                    # A job is a batch, so it names every design it wrote. A worker too old to
                    # batch reports none; it also produced only the first design of the slice,
                    # so say that rather than return a short run as if it were whole.
                    names = [str(x) for x in (row.get("names") or [])]
                    if not names:
                        raise RuntimeError(
                            f"worker returned no design names for job {row.get('id')}: it runs a "
                            "tt-bio older than the one that batches designs per card. Upgrade the "
                            "workers, or run on one card.")
                    for one in names:
                        rows.append({"cif": str(out_dir / f"{one}_generated.cif"), "name": one,
                                     "example_id": row.get("example_id"),
                                     "tokens": row.get("tokens"),
                                     "runtime_s": row.get("runtime_s")})
                    if verbose:
                        print(f"[design:{name}] {len(names)} design(s) done "
                              f"({row.get('runtime_s', '?')}s)", flush=True)
                else:
                    failures[name] = (row.get("error") or "failed").strip()
        status = snap.get("status")
        if status == "canceled":
            print("Run canceled — stopping.", flush=True)
            return []
        if status in ("ok", "failed"):
            break
        time.sleep(0.5)

    for name, err in failures.items():
        print(f"  ✗ {name}: {(err.splitlines() or ['failed'])[0]}", flush=True)
    if not rows and failures:
        raise RuntimeError("Every design failed — nothing was produced. First error:\n"
                           + next(iter(failures.values()), "unknown (see worker logs)"))
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows
