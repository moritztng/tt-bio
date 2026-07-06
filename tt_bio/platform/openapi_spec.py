"""OpenAPI 3.1 contract for the JapanFold ``/v1`` API.

Hand-authored (not generated) so it stays small and readable. It is the single
source of truth a customer can feed to an SDK generator or an
OpenAPI-to-MCP-server tool to get typed clients and agent tools for free.
"""

from __future__ import annotations

_JOB = {
    "type": "object",
    "properties": {
        "object": {"type": "string", "const": "job"},
        "id": {"type": "string"},
        "kind": {"type": "string", "enum": ["predict", "design"]},
        "status": {"type": "string", "enum": ["queued", "running", "succeeded", "failed", "canceled"]},
        "name": {"type": "string"},
        "model": {"type": ["string", "null"]},
        "protocol": {"type": ["string", "null"]},
        "progress": {"type": ["number", "null"], "description": "0..1, or null when indeterminate"},
        "stage": {"type": ["string", "null"]},
        "total": {"type": "integer"},
        "done": {"type": "integer"},
        "error": {"type": ["string", "null"]},
        "created_at": {"type": ["string", "null"], "format": "date-time"},
        "started_at": {"type": ["string", "null"], "format": "date-time"},
        "finished_at": {"type": ["string", "null"], "format": "date-time"},
        "results_ready": {"type": "boolean"},
        "links": {"type": "object"},
    },
    "required": ["object", "id", "kind", "status"],
}

_PROBLEM = {
    "type": "object",
    "description": "RFC 9457 problem detail.",
    "properties": {
        "type": {"type": "string"}, "title": {"type": "string"},
        "status": {"type": "integer"}, "detail": {"type": "string"},
        "instance": {"type": "string"},
    },
}


def _problem_responses(*codes):
    titles = {400: "Invalid request", 401: "Unauthorized", 404: "Not found",
              429: "Too many requests"}
    return {str(c): {"description": titles.get(c, "Error"),
                     "content": {"application/problem+json":
                                 {"schema": {"$ref": "#/components/schemas/Problem"}}}}
            for c in codes}


def build_spec(version: str) -> dict:
    job_ref = {"$ref": "#/components/schemas/Job"}
    json_body = lambda schema: {"required": True, "content": {"application/json": {"schema": schema}}}
    ok_job = {"description": "Job", "content": {"application/json": {"schema": job_ref}}}

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "JapanFold API",
            "version": version,
            "summary": "Biomolecular structure prediction and binder design on Tenstorrent.",
            "description": (
                "Async job API for protein/complex structure prediction (Boltz-2, "
                "ESMFold2, Protenix) and binder design (BoltzGen). Submit a job, poll "
                "its status, then download structures and scores. All endpoints require "
                "an API key (`Authorization: Bearer <key>`)."
            ),
        },
        "servers": [{"url": "https://japanfold.com"}, {"url": "http://localhost:8080"}],
        "security": [{"bearerAuth": []}],
        "tags": [{"name": "discovery"}, {"name": "predictions"}, {"name": "designs"}, {"name": "jobs"}],
        "paths": {
            "/v1/health": {"get": {"tags": ["discovery"], "operationId": "getHealth",
                                   "summary": "Liveness + API version.", "security": [],
                                   "responses": {"200": {"description": "OK"}}}},
            "/v1/models": {"get": {"tags": ["discovery"], "operationId": "listModels",
                                   "summary": "Available models, protocols, parameters and limits.",
                                   "security": [],
                                   "responses": {"200": {"description": "Catalog"}}}},
            "/v1/predictions": {"post": {
                "tags": ["predictions"], "operationId": "createPrediction",
                "summary": "Predict the 3D structure (and affinity) of a protein/complex.",
                "description": "Provide `sequence` (single chain), `input` (one fasta/yaml "
                               "string), or `targets` (a list). Returns a job to poll.",
                "requestBody": json_body({"$ref": "#/components/schemas/PredictRequest"}),
                "responses": {"202": ok_job, **_problem_responses(400, 401, 429)}}},
            "/v1/designs": {"post": {
                "tags": ["designs"], "operationId": "createDesign",
                "summary": "Design binders/proteins against a target (BoltzGen).",
                "requestBody": json_body({"$ref": "#/components/schemas/DesignRequest"}),
                "responses": {"202": ok_job, **_problem_responses(400, 401, 429)}}},
            "/v1/jobs": {"get": {
                "tags": ["jobs"], "operationId": "listJobs", "summary": "List your jobs (paginated).",
                "parameters": [
                    {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20, "maximum": 100}},
                    {"name": "cursor", "in": "query", "schema": {"type": "string"}},
                ],
                "responses": {"200": {"description": "Job list"}, **_problem_responses(401)}}},
            "/v1/jobs/{job_id}": {
                "get": {"tags": ["jobs"], "operationId": "getJob", "summary": "Poll a job's status.",
                        "parameters": [{"name": "job_id", "in": "path", "required": True,
                                        "schema": {"type": "string"}}],
                        "responses": {"200": ok_job, **_problem_responses(401, 404)}},
                "delete": {"tags": ["jobs"], "operationId": "deleteJob", "summary": "Delete a job and its data.",
                           "parameters": [{"name": "job_id", "in": "path", "required": True,
                                           "schema": {"type": "string"}}],
                           "responses": {"200": {"description": "Deleted"}, **_problem_responses(401, 404)}}},
            "/v1/jobs/{job_id}/results": {"get": {
                "tags": ["jobs"], "operationId": "getResults",
                "summary": "List downloadable artifacts and scores once ready.",
                "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Results manifest"}, **_problem_responses(401, 404)}}},
            "/v1/jobs/{job_id}/artifacts/{path}": {"get": {
                "tags": ["jobs"], "operationId": "getArtifact",
                "summary": "Download one structure/score file (CIF/PDB).",
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "path", "in": "path", "required": True, "schema": {"type": "string"}},
                ],
                "responses": {"200": {"description": "File",
                                      "content": {"chemical/x-cif": {}, "chemical/x-pdb": {}}},
                              **_problem_responses(401, 404)}}},
            "/v1/jobs/{job_id}/archive": {"get": {
                "tags": ["jobs"], "operationId": "getArchive",
                "summary": "Download all results as a zip bundle.",
                "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Zip", "content": {"application/zip": {}}},
                              **_problem_responses(401, 404)}}},
            "/v1/jobs/{job_id}/logs": {"get": {
                "tags": ["jobs"], "operationId": "getLogs", "summary": "Plain-text run log.",
                "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Log", "content": {"text/plain": {}}},
                              **_problem_responses(401, 404)}}},
            "/v1/jobs/{job_id}/cancel": {"post": {
                "tags": ["jobs"], "operationId": "cancelJob", "summary": "Cancel a running/queued job.",
                "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Canceled"}, **_problem_responses(401, 404)}}},
        },
        "components": {
            "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer",
                                               "description": "API key: `Authorization: Bearer jf_live_...`"}},
            "schemas": {
                "Job": _JOB,
                "Problem": _PROBLEM,
                "Target": {"type": "object", "properties": {
                    "content": {"type": "string", "description": "FASTA or YAML input."},
                    "name": {"type": "string"}}, "required": ["content"]},
                "PredictRequest": {"type": "object", "properties": {
                    "model": {"type": "string", "enum": ["boltz2", "esmfold2", "esmfold2-fast", "protenix-v2"],
                              "default": "boltz2"},
                    "name": {"type": "string"},
                    "sequence": {"type": "string", "description": "Single protein chain (convenience)."},
                    "input": {"type": "string", "description": "One FASTA/YAML string."},
                    "targets": {"type": "array", "items": {"$ref": "#/components/schemas/Target"}},
                    "params": {"type": "object", "description": "Model params: use_msa_server, fast, "
                               "recycling_steps, sampling_steps, diffusion_samples, output_format."},
                }},
                "DesignRequest": {"type": "object", "properties": {
                    "protocol": {"type": "string", "enum": ["protein-anything", "peptide-anything",
                                 "nanobody-anything", "antibody-anything", "protein-small_molecule",
                                 "protein-redesign"]},
                    "name": {"type": "string"},
                    "spec": {"type": "string", "description": "YAML design spec (target + binder request)."},
                    "params": {"type": "object", "description": "num_designs, budget, fast."},
                }, "required": ["spec"]},
            },
        },
    }
