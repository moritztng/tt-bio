"""Standalone offline-MSA server.

One machine hosts the ColabFold UniRef30 database (~500 GB) and serves unpaired
``{seq_hash}.a3m`` over HTTP; every other machine (and the web portal) fetches
MSAs from it instead of maintaining its own copy of the database. The search
engine is exactly the offline path used by ``tt-bio predict``
(``compute_msa_offline`` -> ``colabfold_search``), wrapped in a small stdlib HTTP
front with a shared on-disk cache and a search-concurrency cap so MSA load can't
oversubscribe the host.

API (JSON over HTTP):

    POST /msa     {"sequences": ["MKT...", ...]}
                  -> {"results": {"<sha256[:16]>": "<a3m text>" | null, ...}}
    GET  /healthz -> "ok"

If the server is started with a token, requests must send
``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_PORT = 8765
# Sequences per colabfold_search call (one batched search, like the CLI offline
# path). Overridable for tuning memory vs per-call size on huge inputs.
SEARCH_CHUNK = max(1, int(os.environ.get("TT_MSA_SEARCH_CHUNK", "2000")))
# Sequences per client HTTP request, so a 40k fetch is many bounded, retryable
# calls rather than one multi-hour connection.
REQUEST_CHUNK = max(1, int(os.environ.get("TT_MSA_REQUEST_CHUNK", "1000")))


def seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


class MsaService:
    """Offline MSA engine with a shared a3m cache and a search-concurrency cap.

    ``resolve`` returns one unpaired a3m per input sequence, searching only the
    ones not already cached. Searches honor ``max_concurrent`` so concurrent
    clients can't spawn an unbounded number of CPU-heavy colabfold_search runs.
    """

    def __init__(self, db_path: str, cache_dir, use_env: bool = False,
                 max_concurrent: int = 1):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_env = use_env
        self._sem = threading.Semaphore(max(1, int(max_concurrent)))

    def _a3m(self, h: str) -> Path:
        return self.cache_dir / f"{h}.a3m"

    def resolve(self, sequences: list[str]) -> dict[str, str | None]:
        from tt_bio.main import compute_msa_offline

        wanted = {seq_hash(s): s for s in sequences if s}
        missing = [(h, s) for h, s in wanted.items() if not self._a3m(h).exists()]
        for i in range(0, len(missing), SEARCH_CHUNK):
            # Re-check existence at search time: a concurrent request may have
            # just cached some of these (compute_msa_offline writes atomically).
            chunk = {h: s for h, s in missing[i:i + SEARCH_CHUNK] if not self._a3m(h).exists()}
            if not chunk:
                continue
            with self._sem:
                compute_msa_offline(chunk, f"msa_server_{i // SEARCH_CHUNK}",
                                    self.cache_dir, self.db_path,
                                    use_env=self.use_env, pair=False)
        out: dict[str, str | None] = {}
        for h in wanted:
            p = self._a3m(h)
            out[h] = p.read_text() if p.exists() else None
        return out


def _make_handler(service: MsaService, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):  # keep the server log quiet
            pass

        def _send(self, code: int, body: bytes = b""):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _authed(self) -> bool:
            return not token or self.headers.get("Authorization", "") == f"Bearer {token}"

        def do_GET(self):
            if self.path.rstrip("/") == "/healthz":
                self._send(200, b'"ok"')
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            if not self._authed():
                self._send(401, b'{"error":"unauthorized"}')
                return
            if self.path.rstrip("/") != "/msa":
                self._send(404, b'{"error":"not found"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                seqs = payload.get("sequences")
                if not isinstance(seqs, list):
                    raise ValueError("'sequences' must be a list")
            except Exception as e:
                self._send(400, json.dumps({"error": f"bad request: {e}"}).encode())
                return
            try:
                results = service.resolve([str(s) for s in seqs])
                self._send(200, json.dumps({"results": results}).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)[:300]}).encode())

    return Handler


def run_server(host: str, port: int, db_path: str, cache_dir, use_env: bool = False,
               max_concurrent: int = 1, token: str | None = None) -> None:
    service = MsaService(db_path, cache_dir, use_env, max_concurrent)
    httpd = ThreadingHTTPServer((host, port), _make_handler(service, token))
    print(f"[msa-server] http://{host}:{port}  db={db_path}  cache={service.cache_dir}  "
          f"use_env={use_env}  max_concurrent={max_concurrent}"
          f"{'  (token required)' if token else ''}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def fetch_msa(sequences, msa_dir, endpoint: str, *, token: str | None = None,
              timeout: float = 86400.0) -> None:
    """Fetch unpaired a3m for ``sequences`` from a tt-bio MSA server and write
    ``{hash}.a3m`` into ``msa_dir`` (the same cache contract as
    ``compute_msa_offline``, so callers resolve MSA identically afterwards).

    ``sequences`` may be a list of sequences or a ``{hash: seq}`` dict. Requests
    are chunked so each HTTP call is bounded and independently retryable.
    """
    seqs = list(sequences.values()) if isinstance(sequences, dict) else list(sequences)
    msa_dir = Path(msa_dir)
    msa_dir.mkdir(parents=True, exist_ok=True)
    url = endpoint.rstrip("/") + "/msa"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for i in range(0, len(seqs), REQUEST_CHUNK):
        body = json.dumps({"sequences": seqs[i:i + REQUEST_CHUNK]}).encode()
        with urlopen(Request(url, data=body, headers=headers, method="POST"), timeout=timeout) as r:
            results = json.loads(r.read()).get("results", {})
        for h, a3m in results.items():
            if a3m:
                tmp = msa_dir / f".{h}.a3m.{os.getpid()}.tmp"
                tmp.write_text(a3m)
                os.replace(tmp, msa_dir / f"{h}.a3m")
