"""Flask app for the ai& Bio platform.

A thin HTTP layer over :class:`tt_bio.platform.jobs.JobManager`. Serves a small
JSON API under ``/api`` and the built React single-page app for everything else.
"""

from __future__ import annotations

import mimetypes
import os
import secrets
from pathlib import Path

from flask import Flask, g, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from .catalog import catalog
from .http_common import client_ip, serve_archive, serve_artifact, serve_log, serve_structure, submit_job
from .jobs import CapacityError, JobManager

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"  # built React app (npm run build output)
_LANDING = _HERE / "landing"  # hand-crafted landing page for the apex domain
# Hosts that get the landing page instead of the SPA. demo.japanfold.com and
# api.japanfold.com are deliberately absent: they keep serving the SPA / API.
# APEX DELIBERATELY EXCLUDED (Moritz, 2026-08-04): landing page stays on
# landing.japanfold.com only (development/preview surface). japanfold.com and
# www.japanfold.com keep serving the SPA. The apex cutover was OWED pending
# Moritz greenlight and was done without it; this restores the intended split.
# Do not re-add the apex here without an explicit greenlight.
_LANDING_HOSTS = frozenset({"landing.japanfold.com"})

# Anonymous per-visitor session. No login: the server mints an unguessable id in
# an HttpOnly cookie on first contact and tags every job with it; a job is only
# ever visible/controllable by the session that created it. Set
# AIAND_BIO_SECURE_COOKIES=1 behind HTTPS so the cookie is TLS-only.
_SESSION_COOKIE = "aiandbio_sid"
_SECURE_COOKIES = os.environ.get("AIAND_BIO_SECURE_COOKIES", "0").lower() not in ("0", "false", "")

mimetypes.add_type("chemical/x-cif", ".cif")
mimetypes.add_type("chemical/x-pdb", ".pdb")


def create_app(workspace: str | os.PathLike | None = None, *,
               cluster=None, max_concurrent: int = 32,
               msa_db_path: str | None = "/data/colabfold_db", msa_mode: str = "auto") -> Flask:
    workspace = workspace or os.environ.get(
        "AIAND_BIO_WORKSPACE", str(Path.home() / ".aiand-bio" / "jobs")
    )
    manager = JobManager(workspace, cluster=cluster, max_concurrent=max_concurrent,
                         msa_db_path=msa_db_path, msa_mode=msa_mode)

    app = Flask(__name__, static_folder=None)
    CORS(app)  # allow the Vite dev server to reach the API in development
    app.config["manager"] = manager
    app.config["cluster"] = cluster
    # Cap request bodies: the demo limits keep real inputs tiny (≤10 small
    # targets), so 8 MB is generous and stops oversized-payload abuse early.
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

    # ---- Anonymous session ------------------------------------------------
    @app.before_request
    def _ensure_session():
        # The /v1 customer API authenticates by key, not cookie — don't mint an
        # anonymous session for those requests (nothing reads it there).
        if request.path.startswith("/v1/"):
            return
        sid = request.cookies.get(_SESSION_COOKIE)
        if not sid or not (16 <= len(sid) <= 128) or not sid.replace("-", "").replace("_", "").isalnum():
            sid = secrets.token_urlsafe(24)  # 192-bit
            g._new_sid = sid
        g.session_id = sid

    @app.after_request
    def _persist_session(resp):
        sid = getattr(g, "_new_sid", None)
        if sid is not None:
            resp.set_cookie(_SESSION_COOKIE, sid, max_age=30 * 24 * 3600,
                            httponly=True, secure=_SECURE_COOKIES, samesite="Lax", path="/")
        return resp

    def _owns(job_id: str) -> bool:
        """True only if the current session owns this job. Used by every per-job
        route to 404 (never 403/200) on someone else's — or a nonexistent — job."""
        return manager.owner_of(job_id) == g.session_id

    # ---- API ----------------------------------------------------------
    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "aiand-bio"})

    @app.get("/api/catalog")
    def get_catalog():
        return jsonify(catalog())

    @app.get("/api/cluster")
    def get_cluster():
        cl = app.config.get("cluster")
        if cl is None:
            return jsonify({
                "enabled": False, "controller_alive": False, "hosts": [],
                "online_workers": 0, "total_workers": 0, "runs": {}, "jobs": {},
            })
        return jsonify(cl.status())

    @app.get("/api/jobs")
    def list_jobs():
        return jsonify({"jobs": manager.list(owner=g.session_id)})

    @app.post("/api/jobs")
    def create_job():
        try:
            body = request.get_json(force=True, silent=True)
        except Exception:
            body = None
        if not isinstance(body, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        try:
            job = submit_job(manager, body, owner=g.session_id, ip=client_ip())
        except CapacityError as e:
            return jsonify({"error": str(e)}), 429
        except (ValueError, KeyError, TypeError) as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(job.to_dict()), 201

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id):
        if not _owns(job_id):
            return jsonify({"error": "not found"}), 404
        d = manager.get(job_id)
        return (jsonify(d), 200) if d else (jsonify({"error": "not found"}), 404)

    @app.get("/api/jobs/<job_id>/log")
    def get_log(job_id):
        if not _owns(job_id):
            return jsonify({"error": "not found"}), 404
        return serve_log(manager, job_id)

    @app.get("/api/jobs/<job_id>/structure/<path:relpath>")
    def get_structure(job_id, relpath):
        if not _owns(job_id):
            return jsonify({"error": "not found"}), 404
        return serve_structure(manager, job_id, relpath) or (jsonify({"error": "not found"}), 404)

    @app.get("/api/jobs/<job_id>/artifact/<path:relpath>")
    def get_artifact(job_id, relpath):
        if not _owns(job_id):
            return jsonify({"error": "not found"}), 404
        return serve_artifact(manager, job_id, relpath) or (jsonify({"error": "not found"}), 404)

    @app.get("/api/jobs/<job_id>/archive")
    def archive_job(job_id):
        if not _owns(job_id):
            return jsonify({"error": "not found"}), 404
        return serve_archive(manager, job_id) or (jsonify({"error": "no results yet"}), 404)

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id):
        if not _owns(job_id):
            return jsonify({"error": "not found"}), 404
        return jsonify({"canceled": manager.cancel(job_id)})

    @app.delete("/api/jobs/<job_id>")
    def delete_job(job_id):
        if not _owns(job_id):
            return jsonify({"error": "not found"}), 404
        return jsonify({"deleted": manager.delete(job_id)})

    # ---- Customer API (/v1, API-key auth) -----------------------------
    # A stable, documented, key-authenticated surface for programmatic and agent
    # use (Claude Science, SDKs, CLI). Additive: it reuses the same JobManager,
    # so the web UI above is unaffected. See api_v1.py / openapi_spec.py.
    from . import api_v1
    api_v1.register(app)

    # ---- Static SPA + apex landing page -------------------------------
    @app.get("/")
    @app.get("/<path:path>")
    def spa(path: str = ""):
        if path.startswith("api/") or path.startswith("v1/"):
            return jsonify({"error": "not found"}), 404
        # The apex domain shows the marketing landing page; the interactive
        # demo SPA lives on demo.japanfold.com. API paths above are identical
        # on every host. Falls through to the SPA when no landing page is
        # deployed (dev boxes, older checkouts).
        host = request.host.partition(":")[0].lower()
        if host in _LANDING_HOSTS:
            candidate = _LANDING / path
            if path and candidate.is_file():
                return send_from_directory(_LANDING, path)
            index = _LANDING / "index.html"
            if index.exists():
                resp = send_file(index)
                resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                return resp
        candidate = _STATIC / path
        if path and candidate.is_file():
            resp = send_from_directory(_STATIC, path)
            # Vite asset filenames are content-hashed, so they're safe to cache
            # forever — a new build produces new names.
            if path.startswith("assets/"):
                resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return resp
        index = _STATIC / "index.html"
        if index.exists():
            resp = send_file(index)
            # index.html names the current hashed bundle, so it must never be
            # cached — otherwise a redeploy needs a manual hard-refresh to show.
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return resp
        return (
            "<h1>ai& Bio</h1><p>Frontend not built. Run "
            "<code>npm install &amp;&amp; npm run build</code> in "
            "<code>tt_bio/platform/frontend</code>.</p>",
            200,
        )

    return app


def serve(host: str = "0.0.0.0", port: int = 8080, workspace: str | None = None,
          debug: bool = False, *, cluster_enabled: bool = True,
          controller_port: int = 8765, controller_bind: str = "0.0.0.0",
          num_devices: int = 0, device_ids: str | None = None,
          accelerator: str = "tenstorrent", max_concurrent: int = 32,
          msa_db_path: str | None = "/data/colabfold_db", msa_mode: str = "auto") -> None:
    workspace = workspace or os.environ.get(
        "AIAND_BIO_WORKSPACE", str(Path.home() / ".aiand-bio" / "jobs")
    )
    cluster = None
    if cluster_enabled:
        from .cluster import Cluster
        cluster = Cluster(
            workspace, enabled=True, bind_host=controller_bind, port=controller_port,
            accelerator=accelerator, num_devices=num_devices, device_ids=device_ids,
        )
        cluster.start()

    app = create_app(workspace, cluster=cluster, max_concurrent=max_concurrent,
                     msa_db_path=msa_db_path, msa_mode=msa_mode)
    print(f"\n  ai& Bio  →  http://{host}:{port}", flush=True)
    if cluster is not None:
        print(f"  fleet controller → {cluster.join_url}", flush=True)
        print(f"  add a galaxy: tt-bio worker --connect {cluster.join_url}\n", flush=True)
    try:
        if debug:
            # Local development only (reloader OFF so it never double-opens devices).
            app.run(host=host, port=port, debug=True, threaded=True, use_reloader=False)
        else:
            # Production: a real WSGI server. The UI polls /api/jobs and /api/cluster
            # every ~2.5s, so many concurrent visitors means many short requests; a
            # thread pool serves them concurrently while the heavy work runs off in
            # the JobManager/cluster. (Flask's dev server is not for production and
            # bottlenecks under that polling load.) waitress handles SIGTERM/SIGINT
            # gracefully and returns, so the cluster.shutdown() below still runs.
            from waitress import serve as _wsgi_serve
            _wsgi_serve(app, host=host, port=port, threads=32,
                        channel_timeout=120, ident="JapanFold")
    finally:
        if cluster is not None:
            cluster.shutdown()
