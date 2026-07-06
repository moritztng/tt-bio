"""JapanFold customer API — ``/v1``.

A stable, documented, API-key-authenticated HTTP surface for programmatic and
agent use (Claude Science, Claude Code, Codex, Gemini CLI, plain scripts). It is
a thin, additive layer over the same :class:`~tt_bio.platform.jobs.JobManager`
that backs the web UI: a customer's API key resolves to a customer id, used as
the job ``owner`` — so isolation, listing and per-customer throttling come for
free.

Model (async, mirroring Replicate / Boltz): ``POST /v1/predictions`` and
``POST /v1/designs`` create a job and return ``202``; ``GET /v1/jobs/{id}``
polls; ``GET /v1/jobs/{id}/results`` lists downloadable artifacts. A caller may
send ``Prefer: wait[=seconds]`` to have a create/poll request block until the
job finishes (up to a cap) and return the terminal job in one round-trip.
Errors are RFC 9457 problem+json; the contract is published at
``GET /v1/openapi.json``.
"""

from __future__ import annotations

import functools
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone

from flask import Blueprint, current_app, g, jsonify, request

from . import apikeys
from .catalog import catalog
from .http_common import client_ip, serve_archive, serve_log, serve_structure, submit_job
from .jobs import CANCELED, CapacityError, FAILED, SUCCEEDED

API_VERSION = "1.0.0"
TERMINAL = {SUCCEEDED, FAILED, CANCELED}  # canonical lifecycle vocabulary (jobs.py)

# `Prefer: wait` synchronous-hold bounds (seconds). Default hold if `wait` has no
# value; hard cap regardless of what the client asks for (protects worker threads).
_WAIT_DEFAULT = 25
_WAIT_MAX = 60
_WAIT_POLL = 1.0

bp = Blueprint("v1", __name__, url_prefix="/v1")

# Endpoints reachable without a key (discovery + liveness + contract).
_PUBLIC_ENDPOINTS = {"v1.health", "v1.openapi", "v1.models"}

# Small bounded idempotency cache: (customer, Idempotency-Key) -> job_id, so a
# retried POST returns the same job instead of launching a duplicate run.
_IDEMPOTENCY_MAX = 2048
_idem: "OrderedDict[tuple[str, str], str]" = OrderedDict()
_idem_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _mgr():
    return current_app.config["manager"]


def _problem(status, title, detail=None, type_="about:blank", **ext):
    """An RFC 9457 problem+json response."""
    body = {"type": type_, "title": title, "status": status, "instance": request.path}
    if detail:
        body["detail"] = detail
    body.update(ext)
    resp = jsonify(body)
    resp.status_code = status
    resp.headers["Content-Type"] = "application/problem+json"
    return resp


def _iso(ts):
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat() if ts else None
    except (TypeError, ValueError, OSError):
        return None


def _shape(d: dict) -> dict:
    """Reshape a JobManager dict into the stable public ``job`` resource."""
    jid = d.get("id")
    base = f"/v1/jobs/{jid}"
    return {
        "object": "job", "id": jid,
        "kind": d.get("kind"), "status": d.get("status"), "name": d.get("name"),
        "model": d.get("model"), "protocol": d.get("protocol"),
        "progress": d.get("progress"), "stage": d.get("stage"),
        "total": d.get("total"), "done": d.get("done"),
        "params": d.get("params") or {}, "error": d.get("error"),
        "created_at": _iso(d.get("created_at")), "started_at": _iso(d.get("started_at")),
        "finished_at": _iso(d.get("finished_at")),
        "results_ready": bool(d.get("results_ready") or (d.get("results") or {}).get("ready")),
        "links": {"self": base, "results": f"{base}/results",
                  "archive": f"{base}/archive", "logs": f"{base}/logs"},
    }


def _artifacts(job_id: str, results: dict) -> list[dict]:
    """Flatten parsed results into a list of downloadable artifacts (URLs)."""
    base = f"/v1/jobs/{job_id}/artifacts/"
    if results.get("kind") == "predict":
        return [{"type": "structure", "target": target, "path": name, "url": base + name}
                for target, files in (results.get("structures") or {}).items() for name in files]
    if results.get("kind") == "design":
        return [{"type": "structure", "rank": d.get("final_rank"),
                 "path": d["structure"], "url": base + d["structure"]}
                for d in (results.get("designs") or []) if d.get("structure")]
    return []


def _wait_seconds():
    """Parse ``Prefer: wait[=N]``. Returns the hold in seconds, or 0 if absent."""
    prefer = request.headers.get("Prefer", "")
    for part in (p.strip() for p in prefer.split(",")):
        if part == "wait":
            return _WAIT_DEFAULT
        if part.startswith("wait="):
            try:
                return max(0, min(_WAIT_MAX, int(part[5:])))
            except ValueError:
                return _WAIT_DEFAULT
    return 0


def _await_terminal(job_id: str, seconds: int):
    """Block up to ``seconds`` for the job to reach a terminal state (best-effort).
    Polls the cheap in-memory status — no per-iteration results parse."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = _mgr().peek_status(job_id)
        if status is None or status in TERMINAL:
            return
        time.sleep(_WAIT_POLL)


def _job_view(job_id: str):
    """A shaped job resource, honouring ``Prefer: wait`` — hold for a terminal
    state, then return the light status view. Shared by create and GET."""
    wait = _wait_seconds()
    if wait:
        _await_terminal(job_id, wait)
    return _shape(_mgr().brief(job_id))


def owned(fn):
    """Route guard: 404 unless the caller's key owns ``job_id`` (existence is
    never confirmed to a non-owner). Every per-job route wears this."""
    @functools.wraps(fn)
    def wrapper(job_id, *args, **kwargs):
        if _mgr().owner_of(job_id) != g.customer:
            return _problem(404, "Not found", "No such job.")
        return fn(job_id, *args, **kwargs)
    return wrapper


# --------------------------------------------------------------------------- #
# Auth (blueprint-scoped: runs only for /v1 routes)
# --------------------------------------------------------------------------- #
@bp.before_request
def _authenticate():
    if request.endpoint in _PUBLIC_ENDPOINTS or request.method == "OPTIONS":
        return None
    auth = request.headers.get("Authorization", "")
    key = auth[7:].strip() if auth[:7].lower() == "bearer " else request.headers.get("X-API-Key", "")
    customer = apikeys.verify(key)
    if not customer:
        return _problem(401, "Unauthorized",
                        "Provide a valid key as 'Authorization: Bearer <key>' or 'X-API-Key'.",
                        type_="https://japanfold.com/errors/unauthorized")
    g.customer = customer  # becomes the JobManager owner for every downstream call


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
@bp.get("/health")
def health():
    return jsonify({"status": "ok", "service": "japanfold", "api_version": API_VERSION})


@bp.get("/models")
def models():
    c = catalog()
    return jsonify({k: c[k] for k in ("models", "protocols", "predict_params",
                                      "design_params", "limits")} | {"notes": c.get("demo_note")})


@bp.get("/openapi.json")
def openapi():
    from .openapi_spec import build_spec
    return jsonify(build_spec(API_VERSION))


# --------------------------------------------------------------------------- #
# Job creation
# --------------------------------------------------------------------------- #
def _submit(payload: dict):
    """Shared create path: idempotency, submit, Prefer: wait, problem+json errors."""
    mgr = _mgr()
    idem = request.headers.get("Idempotency-Key")
    idem_key = (g.customer, idem) if idem else None
    if idem_key:
        with _idem_lock:
            prior = _idem.get(idem_key)
        if prior and mgr.owner_of(prior) == g.customer:  # replay: return the same job
            return _respond(prior, created=False)
    try:
        job = submit_job(mgr, payload, owner=g.customer, ip=client_ip(), via="v1")
    except CapacityError as e:
        resp = _problem(429, "Too Many Requests", str(e), type_="https://japanfold.com/errors/capacity")
        resp.headers["Retry-After"] = "30"
        return resp
    except (ValueError, KeyError, TypeError) as e:
        return _problem(400, "Invalid request", str(e), type_="https://japanfold.com/errors/invalid-input")
    if idem_key:
        with _idem_lock:
            _idem[idem_key] = job.id
            if len(_idem) > _IDEMPOTENCY_MAX:  # grows one per insert; never exceeds by more
                _idem.popitem(last=False)
    return _respond(job.id, created=True)


def _respond(job_id: str, *, created: bool):
    """Render a job, honouring ``Prefer: wait`` (block for a terminal state)."""
    shaped = _job_view(job_id)
    resp = jsonify(shaped)
    terminal = shaped["status"] in TERMINAL
    resp.status_code = 200 if (terminal or not created) else 202
    resp.headers["Location"] = f"/v1/jobs/{job_id}"
    return resp


def _targets_from(body: dict) -> list[dict]:
    """Normalize the accepted input shapes to the engine's ``[{content, name?}]``.

    - ``targets``: list of ``{content, name?}`` or bare fasta/yaml strings
    - ``input``:   a single fasta/yaml string
    - ``sequence``: a bare protein sequence (wrapped as a one-chain fasta)
    """
    if body.get("targets") is not None:
        out = []
        for i, t in enumerate(body["targets"]):
            if isinstance(t, str):
                out.append({"content": t})
            elif isinstance(t, dict) and isinstance(t.get("content"), str):
                out.append({"content": t["content"], **({"name": t["name"]} if t.get("name") else {})})
            else:
                raise ValueError(f"targets[{i}] must be a string or an object with a 'content' string")
        return out
    if isinstance(body.get("input"), str):
        return [{"content": body["input"]}]
    if isinstance(body.get("sequence"), str):
        return [{"content": f">A|protein\n{body['sequence'].strip()}"}]
    raise ValueError("provide 'targets', 'input', or 'sequence'")


def _json_body():
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        raise ValueError("Body must be a JSON object.")
    return body


@bp.post("/predictions")
def create_prediction():
    try:
        body = _json_body()
        targets = _targets_from(body)
    except ValueError as e:
        return _problem(400, "Invalid request", str(e), type_="https://japanfold.com/errors/invalid-input")
    return _submit({"kind": "predict", "model": body.get("model", "boltz2"),
                    "name": body.get("name"), "targets": targets, "params": body.get("params") or {}})


@bp.post("/designs")
def create_design():
    try:
        body = _json_body()
    except ValueError as e:
        return _problem(400, "Invalid request", str(e), type_="https://japanfold.com/errors/invalid-input")
    spec = body.get("spec") if isinstance(body.get("spec"), str) else body.get("input")
    if not isinstance(spec, str) or not spec.strip():
        return _problem(400, "Invalid request", "Provide a 'spec' (YAML design spec) string.",
                        type_="https://japanfold.com/errors/invalid-input")
    return _submit({"kind": "design", "protocol": body.get("protocol"),
                    "name": body.get("name"), "spec": spec, "params": body.get("params") or {}})


# --------------------------------------------------------------------------- #
# Job read / control
# --------------------------------------------------------------------------- #
@bp.get("/jobs")
def list_jobs():
    jobs = _mgr().list(owner=g.customer)  # newest first
    try:
        limit = max(1, min(100, int(request.args.get("limit", 20))))
    except ValueError:
        limit = 20
    cursor = request.args.get("cursor")
    ids = [j["id"] for j in jobs]
    start = ids.index(cursor) + 1 if cursor in ids else 0
    page = jobs[start:start + limit]
    has_more = start + limit < len(jobs)
    return jsonify({"object": "list", "data": [_shape(j) for j in page],
                    "has_more": has_more,
                    "next_cursor": page[-1]["id"] if has_more else None})


@bp.get("/jobs/<job_id>")
@owned
def get_job(job_id):
    return jsonify(_job_view(job_id))


@bp.get("/jobs/<job_id>/results")
@owned
def get_results(job_id):
    d = _mgr().get(job_id)
    results = d.get("results") or {}
    if not results.get("ready"):
        return jsonify({"object": "results", "job_id": job_id, "ready": False, "status": d.get("status")})
    out = {"object": "results", "job_id": job_id, "ready": True, "kind": results.get("kind"),
           "artifacts": _artifacts(job_id, results), "archive_url": f"/v1/jobs/{job_id}/archive"}
    if results.get("kind") == "predict":
        out["rows"] = results.get("rows")
    else:
        out["designs"] = results.get("designs")
    return jsonify(out)


@bp.get("/jobs/<job_id>/artifacts/<path:relpath>")
@owned
def get_artifact(job_id, relpath):
    return serve_structure(_mgr(), job_id, relpath) or _problem(404, "Not found", "No such artifact.")


@bp.get("/jobs/<job_id>/archive")
@owned
def get_archive(job_id):
    return serve_archive(_mgr(), job_id) or _problem(404, "Not found", "No results yet.")


@bp.get("/jobs/<job_id>/logs")
@owned
def get_logs(job_id):
    return serve_log(_mgr(), job_id)


@bp.post("/jobs/<job_id>/cancel")
@owned
def cancel_job(job_id):
    return jsonify({"canceled": _mgr().cancel(job_id), "id": job_id})


@bp.delete("/jobs/<job_id>")
@owned
def delete_job(job_id):
    return jsonify({"deleted": _mgr().delete(job_id), "id": job_id})


def register(app):
    """Attach the /v1 API to an existing Flask app."""
    app.register_blueprint(bp)
