"""Flask app for the ai& Bio platform.

A thin HTTP layer over :class:`tt_bio.platform.jobs.JobManager`. Serves a small
JSON API under ``/api`` and the built React single-page app for everything else.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from .catalog import catalog
from .jobs import JobManager

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"  # built React app (npm run build output)

mimetypes.add_type("chemical/x-cif", ".cif")
mimetypes.add_type("chemical/x-pdb", ".pdb")


def create_app(workspace: str | os.PathLike | None = None) -> Flask:
    workspace = workspace or os.environ.get(
        "AIAND_BIO_WORKSPACE", str(Path.home() / ".aiand-bio" / "jobs")
    )
    manager = JobManager(workspace)

    app = Flask(__name__, static_folder=None)
    CORS(app)  # allow the Vite dev server to reach the API in development
    app.config["manager"] = manager
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # room for bulk uploads

    # ---- API ----------------------------------------------------------
    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "aiand-bio"})

    @app.get("/api/catalog")
    def get_catalog():
        return jsonify(catalog())

    @app.get("/api/jobs")
    def list_jobs():
        return jsonify({"jobs": manager.list()})

    @app.post("/api/jobs")
    def create_job():
        try:
            job = manager.submit(request.get_json(force=True) or {})
        except (ValueError, KeyError) as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(job.to_dict()), 201

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id):
        d = manager.get(job_id)
        return (jsonify(d), 200) if d else (jsonify({"error": "not found"}), 404)

    @app.get("/api/jobs/<job_id>/log")
    def get_log(job_id):
        if manager.jobs.get(job_id) is None:
            return jsonify({"error": "not found"}), 404
        return app.response_class(manager.log_text(job_id), mimetype="text/plain")

    @app.get("/api/jobs/<job_id>/structure/<path:relpath>")
    def get_structure(job_id, relpath):
        path = manager.structure_file(job_id, relpath)
        if not path:
            return jsonify({"error": "not found"}), 404
        return send_file(path, as_attachment=False, download_name=path.name)

    @app.get("/api/jobs/<job_id>/archive")
    def archive_job(job_id):
        path = manager.archive(job_id)
        if not path:
            return jsonify({"error": "no results yet"}), 404
        return send_file(path, as_attachment=True, download_name=f"{job_id}-results.zip")

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id):
        return jsonify({"canceled": manager.cancel(job_id)})

    @app.delete("/api/jobs/<job_id>")
    def delete_job(job_id):
        return jsonify({"deleted": manager.delete(job_id)})

    # ---- Static SPA ---------------------------------------------------
    @app.get("/")
    @app.get("/<path:path>")
    def spa(path: str = ""):
        if path.startswith("api/"):
            return jsonify({"error": "not found"}), 404
        candidate = _STATIC / path
        if path and candidate.is_file():
            return send_from_directory(_STATIC, path)
        index = _STATIC / "index.html"
        if index.exists():
            return send_file(index)
        return (
            "<h1>ai& Bio</h1><p>Frontend not built. Run "
            "<code>npm install &amp;&amp; npm run build</code> in "
            "<code>tt_bio/platform/frontend</code>.</p>",
            200,
        )

    return app


def serve(host: str = "0.0.0.0", port: int = 8080, workspace: str | None = None,
          debug: bool = False) -> None:
    app = create_app(workspace)
    banner = f"\n  ai& Bio  →  http://{host}:{port}\n"
    print(banner, flush=True)
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
