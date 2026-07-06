"""Helpers shared by the two HTTP surfaces over :class:`JobManager` — the web
UI's ``/api`` (session cookies) and the customer ``/v1`` API (API keys).

Keeping the client-IP logic and the submit→log sequence here means there is
exactly one implementation of each, used by both surfaces.
"""

from __future__ import annotations

from flask import request

from .jobs import CapacityError, Job, JobManager


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
