"""Fleet dispatch for PXDesign: shard a target's designs across a controller's workers.

The local path is :func:`run_design`, called both by ``tt-bio design --model pxdesign``
and by each worker for its own shard, so there is exactly one place that turns a target
YAML into binder CIFs.

Sharding is ONE JOB PER DESIGN, unlike RFD3's one-job-per-spec. RFD3 keeps a spec's
designs together because they share the featurize + TokenInitializer pass and batch
bit-identically; PXDesign has no such shared prefix worth protecting. Measured on the
serving pool at 592 tokens: 0.1186 s per diffusion step and 10.8 s of fixed
per-process cost (import, 556 MB checkpoint load, featurise, write). Four designs at
`n_step` 200 batched into one shard is ~105 s on one card; the same four fanned out one
per card is ~34 s. The batch axis is worth at most 1.25x (main.py's own --num_designs
help), against up to Nx from the fan-out, so the fan-out wins and each design gets its
own seed rather than a slice of one batched trajectory.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import torch


def run_design(inputs: Path, out_dir, cache, num_designs: int, n_step: int, seed: int,
               *, stem: str | None = None, verbose: bool = True) -> list[dict]:
    """Design ``num_designs`` binder backbones against the target in ``inputs``.

    Returns write_design_cifs' rows: one per design, carrying the CIF path, the binder
    residue/atom counts and ``fit_rmsd`` — the residual of fitting the model's own
    reconstruction of the target onto the real target, which is the end-to-end
    correctness signal (a broken conditioning path lands in the tens of angstroms).
    """
    from tt_bio.main import ensure_p300_mesh_descriptor, ensure_pxdesign_weights
    from tt_bio.pxdesign.inputs import design_inputs_from_yaml
    from tt_bio.pxdesign.model import ProtenixDesign
    from tt_bio.pxdesign.write import write_design_cifs

    inputs = Path(inputs)
    torch.set_grad_enabled(False)
    feats = design_inputs_from_yaml(inputs)
    # The model's parameters are float32 and the featurizer builds integer bins and
    # masks; the harness converts once before the forward and so does this.
    feats = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v)
             for k, v in feats.items()}

    ckpt = ensure_pxdesign_weights(Path(cache).expanduser())
    ensure_p300_mesh_descriptor()

    if verbose:
        n_token = int(feats["restype"].shape[0])
        print(f"Designing {num_designs} binder(s) against {inputs.name} "
              f"({n_token} tokens, {n_step} steps) → {out_dir}", flush=True)
    model = ProtenixDesign.load_from_checkpoint(str(ckpt))
    coords = model.design(feats, n_step=n_step, n_sample=num_designs, seed=seed)
    rows = write_design_cifs(coords, feats, out_dir, stem=stem or inputs.stem)
    _write_metrics(out_dir, rows)
    return rows


def _write_metrics(out_dir, rows: list[dict]) -> None:
    """Drop the per-design numbers beside the CIFs as `designs.json`.

    A binder backbone CIF carries no sequence, no B-factor and no confidence, so
    everything worth showing about a design lives in these rows — `fit_rmsd` above all.
    Without a file they exist only on stdout, which anything reading the output
    directory (the platform's results view, a user's script) cannot use.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prior = []
    path = out / "designs.json"
    if path.exists():                       # fleet shards land one at a time
        try:
            prior = [r for r in json.loads(path.read_text()) if isinstance(r, dict)]
        except Exception:
            prior = []
    keep = ("fit_rmsd", "binder_residues", "binder_atoms", "conditioned_tokens")
    fresh = [{"id": Path(str(r["cif"])).stem, **{k: r[k] for k in keep if k in r}}
             for r in rows]
    have = {r["id"] for r in fresh}
    merged = [r for r in prior if r.get("id") not in have] + fresh
    merged.sort(key=lambda r: str(r.get("id")))
    path.write_text(json.dumps(merged, indent=2))


def run_design_via_controller(
    inputs: Path,
    out_dir,
    *,
    controller_url: str,
    num_designs: int = 1,
    n_step: int = 400,
    seed: int = 42,
    run_id: str | None = None,
    owner: str | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Fleet twin of :func:`run_design`: one shard per design, collected here.

    The target YAML and the structure it names are both shipped inline, so workers on
    other machines need no shared filesystem. Each shard runs in-process on its
    worker's already-open chip (no cold device open) and gets seed ``seed + k``, which
    is the same per-design seeding the local non-batched path uses.
    """
    from tt_bio.distributed import connect_controller
    from tt_bio.main import _write_job_outputs
    from tt_bio.pxdesign.inputs import read_design_yaml

    inputs = Path(inputs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the target through the engine's own parser, so a malformed YAML fails here
    # with the same message the local path gives rather than N times on N workers.
    spec = read_design_yaml(inputs)          # raises on a missing target.file
    src = Path(str(spec["structure"]))

    client, online = connect_controller(controller_url)

    config = {
        "kind": "design",
        "engine": "pxdesign",
        "n_step": int(n_step),
        "target": {"name": inputs.name, "content": inputs.read_text()},
        "structures": [{"name": src.name, "content": src.read_text()}],
    }
    jobs = [{"id": f"pxdesign_{k}", "name": f"design_{k}",
             "input_b64": base64.b64encode(
                 json.dumps({"stem": f"design_{k}", "seed": int(seed) + k}).encode()).decode()}
            for k in range(int(num_designs))]
    run_payload = {"data": "pxdesign-design", "out_dir": str(out_dir),
                   "result_dir": str(out_dir), "jobs": jobs, "config": config,
                   "owner": owner}
    if run_id:
        run_payload["run_id"] = run_id
    run_id = client.create_run(run_payload)["run_id"]
    if verbose:
        print(f"Dispatched {len(jobs)} design shard(s) across {online} worker(s) "
              f"on the fleet at {controller_url}", flush=True)

    rows: list[dict] = []
    failures: dict[str, str] = {}
    after = 0
    announced = False
    last_beat = 0.0
    while True:
        snap = client.events(run_id, after)
        for ev in snap.get("events", []):
            after = max(after, int(ev.get("seq", after)))
            etype = ev.get("event")
            if etype == "start" and not announced:
                announced = True
                # Echo the stage word so log-tailing progress (the platform's stall
                # watchdog, and its progress bar) advances — same as the RFD3 client.
                print("stage: design", flush=True)
            elif etype == "progress":
                now = time.time()
                if verbose and now - last_beat >= 30:
                    last_beat = now
                    print(f"[design:{ev.get('name', '?')}] diffusing… "
                          f"{int(ev.get('elapsed_s') or 0)}s elapsed", flush=True)
            elif etype == "done":
                row = ev.get("row") or {}
                name = str(row.get("name") or row.get("id", ""))
                if row.get("status") == "ok":
                    _write_job_outputs(client, run_id, str(row.get("id")), out_dir)
                    p = out_dir / f"{name}.cif"
                    if p.exists():
                        rows.append({"cif": str(p), "fit_rmsd": row.get("fit_rmsd"),
                                     "binder_residues": row.get("binder_residues"),
                                     "binder_atoms": row.get("binder_atoms"),
                                     "conditioned_tokens": row.get("conditioned_tokens")})
                    if verbose:
                        print(f"[design:{name}] done ({row.get('runtime_s', '?')}s)", flush=True)
                else:
                    failures[name] = (row.get("error") or "failed").strip()
        st = snap.get("status")
        if st == "canceled":
            print("Run canceled — stopping.", flush=True)
            return []
        if st in ("ok", "failed"):
            break
        time.sleep(0.5)

    if failures:
        for name, err in failures.items():
            print(f"  ✗ {name}: {(err.splitlines() or ['failed'])[0]}", flush=True)
    if not rows and failures:
        raise RuntimeError(
            "Every design shard failed — no designs were produced. First error:\n"
            + next(iter(failures.values()), "unknown (see worker logs)"))
    _write_metrics(out_dir, rows)
    return rows
