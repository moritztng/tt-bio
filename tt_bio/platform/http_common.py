"""Helpers shared by the two HTTP surfaces over :class:`JobManager` — the web
UI's ``/api`` (session cookies) and the customer ``/v1`` API (API keys).

Keeping the client-IP logic and the submit→log sequence here means there is
exactly one implementation of each, used by both surfaces.
"""

from __future__ import annotations

from flask import current_app, request, send_file

from .jobs import CapacityError, Job, JobManager


# --- job artifact serving (identical across /api and /v1) -------------------
# Each returns a ready-to-return Flask response, or None when the artifact
# doesn't exist yet — the caller renders its own not-found envelope
# (``{"error": …}`` for /api, RFC 9457 problem+json for /v1).
def serve_structure(manager: JobManager, job_id: str, relpath: str):
    path = manager.structure_file(job_id, relpath)
    return send_file(path, as_attachment=False, download_name=path.name) if path else None


def serve_artifact(manager: JobManager, job_id: str, relpath: str):
    """Serve a job's result file, routed by kind: predict/design artifacts live
    in a nested structures dir (structure_file); embed's manifest.json/*.npz/
    embeddings.parquet live directly in the results dir (artifact_file)."""
    path = (manager.artifact_file(job_id, relpath) if manager.kind_of(job_id) == "embed"
           else manager.structure_file(job_id, relpath))
    return send_file(path, as_attachment=False, download_name=path.name) if path else None


def serve_archive(manager: JobManager, job_id: str):
    path = manager.archive(job_id)
    return send_file(path, as_attachment=True, download_name=f"{job_id}-results.zip") if path else None


def serve_log(manager: JobManager, job_id: str):
    """The run log as text/plain (empty body if the job has produced none yet)."""
    return current_app.response_class(manager.log_text(job_id), mimetype="text/plain")


def client_ip() -> str:
    """The real client IP, for per-IP flood limits and logging.

    Behind Cloudflare (production ingress) the trustworthy source is
    ``CF-Connecting-IP`` (set at the edge, un-spoofable); fall back to the
    rightmost ``X-Forwarded-For`` entry, then the socket peer.
    """
    cf = request.headers.get("CF-Connecting-IP", "").strip()
    if cf:
        return cf
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.remote_addr or "?"


def submit_job(manager: JobManager, payload: dict, *, owner: str, ip: str,
               via: str | None = None) -> Job:
    """Submit a job and record the outcome to the event log.

    Raises :class:`CapacityError` (→ 429) or ``ValueError``/``KeyError``/
    ``TypeError`` (→ 400) on rejection; the caller renders the error in its own
    envelope (``{"error": …}`` for ``/api``, problem+json for ``/v1``).
    """
    sh, iph = manager.evhash(owner), manager.evhash(ip)
    tag = {"via": via} if via else {}
    try:
        job = manager.submit(payload, owner=owner, client_ip=ip)
    except CapacityError as e:
        manager.log_event("job_rejected", reason="capacity", detail=str(e)[:160],
                          session=sh, ip=iph, **tag)
        raise
    except (ValueError, KeyError, TypeError) as e:
        manager.log_event("job_rejected", reason="invalid", detail=str(e)[:160],
                          session=sh, ip=iph, **tag)
        raise
    manager.log_event("job_submitted", job=job.id, kind=job.kind, model=job.model,
                      protocol=job.protocol, total=job.total or None,
                      num_designs=(job.params or {}).get("num_designs"),
                      fast=bool((job.params or {}).get("fast")),
                      session=sh, ip=iph, **tag)
    return job
