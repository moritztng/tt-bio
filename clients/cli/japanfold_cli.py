#!/usr/bin/env python3
"""JapanFold CLI — biomolecular structure prediction & binder design.

A dependency-free (stdlib-only) client for the JapanFold ``/v1`` API. Designed
to be driven both by humans and by coding agents (Claude Code / Claude Science,
Codex, Gemini CLI): every command takes ``--json`` for machine output, auth
comes from an env var for sandboxed agents, and ``predict --wait`` / ``download``
run as a single foreground long-running command (submit → poll → download), so
an agent never has to background a process.

No key is needed for the free public demo (same limits as the web app). An
optional API key raises those limits once you have one.

Auth resolution order:  --api-key  >  $JAPANFOLD_API_KEY  >  ~/.config/japanfold/config.json
Base URL resolution:     --base-url >  $JAPANFOLD_BASE_URL  >  https://api.japanfold.com

Quick start (no key required):
    japanfold predict --sequence MKTAYIAKQR... --wait --out ./out
    japanfold design spec.yaml --protocol nanobody-anything --wait --out ./out
    japanfold embed --sequence MKTAYIAKQR... --wait --out ./out
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

__version__ = "1.0.0"

DEFAULT_BASE_URL = "https://api.japanfold.com"
CONFIG_PATH = Path(os.environ.get("JAPANFOLD_CONFIG",
                                  Path.home() / ".config" / "japanfold" / "config.json"))
TERMINAL = {"succeeded", "failed", "canceled"}


# --------------------------------------------------------------------------- #
# Config / auth
# --------------------------------------------------------------------------- #
_config_cache: dict | None = None


def _load_config() -> dict:
    # Read once per process: resolve_api_key + resolve_base_url both consult it
    # for a single command. _save_config invalidates it.
    global _config_cache
    if _config_cache is None:
        try:
            _config_cache = json.loads(CONFIG_PATH.read_text())
        except (OSError, ValueError):
            _config_cache = {}
    return _config_cache


def _save_config(cfg: dict) -> None:
    global _config_cache
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    tmp.chmod(0o600)
    os.replace(tmp, CONFIG_PATH)
    _config_cache = None  # next read reflects the write


def resolve_api_key(args) -> str | None:
    return getattr(args, "api_key", None) or os.environ.get("JAPANFOLD_API_KEY") \
        or _load_config().get("api_key")


def resolve_base_url(args) -> str:
    return (getattr(args, "base_url", None) or os.environ.get("JAPANFOLD_BASE_URL")
            or _load_config().get("base_url") or DEFAULT_BASE_URL).rstrip("/")


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
# Meaningful exit codes so an agent can branch on the failure mode without
# parsing stderr. 0=ok; 1=job failed / generic; 2=auth; 3=invalid input/usage;
# 4=not found; 5=rate-limited/capacity; 6=unreachable; 130=interrupted.
EXIT_OK, EXIT_ERROR, EXIT_AUTH, EXIT_INPUT, EXIT_NOTFOUND, EXIT_RATE, EXIT_UNREACHABLE = 0, 1, 2, 3, 4, 5, 6
_STATUS_EXIT = {0: EXIT_UNREACHABLE, 400: EXIT_INPUT, 401: EXIT_AUTH, 403: EXIT_AUTH,
                404: EXIT_NOTFOUND, 422: EXIT_INPUT, 429: EXIT_RATE}


class Done(Exception):
    """Raised by a handler to exit with a specific code (e.g. a failed job)."""
    def __init__(self, code):
        self.code = code


class ApiError(Exception):
    def __init__(self, status, body):
        self.status = status
        self.body = body
        if isinstance(body, dict):
            msg = body.get("title") or body.get("error") or str(body)
        else:
            msg = str(body)
        detail = body.get("detail") if isinstance(body, dict) else None
        super().__init__(f"{status}: {msg}" + (f" — {detail}" if detail else ""))

    @property
    def exit_code(self):
        return _STATUS_EXIT.get(self.status, EXIT_ERROR)


def _request(method, base, path, key, *, body=None, raw=False, timeout=120, idempotency=None):
    url = base + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    # A real User-Agent: urllib's default ("Python-urllib/x.y") trips edge bot
    # filters (e.g. Cloudflare 1010) that a named client clears.
    req.add_header("User-Agent", f"japanfold-cli/{__version__}")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if idempotency:
        req.add_header("Idempotency-Key", idempotency)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            if raw:
                return resp.status, payload, dict(resp.headers)
            return resp.status, (json.loads(payload) if payload else {}), dict(resp.headers)
    except urllib.error.HTTPError as e:
        payload = e.read()
        try:
            parsed = json.loads(payload)
        except ValueError:
            parsed = {"title": payload.decode(errors="replace")[:200] or e.reason, "status": e.code}
        raise ApiError(e.code, parsed) from None
    except urllib.error.URLError as e:
        raise ApiError(0, {"title": f"cannot reach {base}", "detail": str(e.reason)}) from None


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _emit(obj, as_json):
    if as_json:
        print(json.dumps(obj, indent=2))


def _err(msg):
    print(f"japanfold: {msg}", file=sys.stderr)


def _progress(job):
    p = job.get("progress")
    bar = f"{int(p * 100):3d}%" if isinstance(p, (int, float)) else "  · "
    stage = job.get("stage") or ""
    return f"[{bar}] {job.get('status'):9} {stage}".rstrip()


# --------------------------------------------------------------------------- #
# Input normalization
# --------------------------------------------------------------------------- #
def _read_input(positional, sequence, inp):
    """Return a request field dict: {sequence|input|...}. Precedence: --sequence,
    --input, then the positional (a file path or '-' for stdin)."""
    if sequence:
        return {"sequence": sequence}
    if inp:
        return {"input": inp}
    if positional:
        if positional == "-":
            return {"input": sys.stdin.read()}
        p = Path(positional)
        if p.is_file():
            return {"input": p.read_text()}
        # Not a file: treat a bare alphabetic token as a sequence, else raw input.
        if positional.replace("\n", "").isalpha():
            return {"sequence": positional}
        return {"input": positional}
    return {}


def _collect_params(pairs, **explicit):
    params = {}
    for k, v in explicit.items():
        if v is not None:
            params[k] = v
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param must be key=value, got: {pair}")
        k, v = pair.split("=", 1)
        # coerce simple scalars
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        params[k] = v
    return params


# --------------------------------------------------------------------------- #
# Poll + download
# --------------------------------------------------------------------------- #
def _poll_until_done(base, key, job_id, *, quiet=False, interval=5.0, max_interval=15.0):
    last = None
    while True:
        _, job, _ = _request("GET", base, f"/v1/jobs/{job_id}", key)
        line = _progress(job)
        if not quiet and line != last:
            print(f"  {job_id[:8]}  {line}", file=sys.stderr)
            last = line
        if job.get("status") in TERMINAL:
            return job
        time.sleep(interval)
        interval = min(max_interval, interval * 1.3)


def _download(base, key, job_id, out_dir, *, name=None):
    """Fetch the results bundle and extract it. Resume-friendly: skips if already
    extracted. Returns the output directory path."""
    dest = Path(out_dir) / (name or job_id)
    marker = dest / ".japanfold-complete"
    if marker.exists():
        return dest
    try:
        _, blob, _ = _request("GET", base, f"/v1/jobs/{job_id}/archive", key, raw=True)
    except ApiError as e:
        raise SystemExit(f"no results to download: {e}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for m in z.namelist():
            # Skip engine bookkeeping files; keep structures + results.json only.
            if m.endswith((".lock", ".tmp", ".bak")):
                continue
            z.extract(m, dest)
    # Persist the final job record + the /v1 results manifest under distinct names
    # so they never clobber the engine's own results.json inside the bundle.
    _, job, _ = _request("GET", base, f"/v1/jobs/{job_id}", key)
    (dest / "job.json").write_text(json.dumps(job, indent=2))
    try:
        _, res, _ = _request("GET", base, f"/v1/jobs/{job_id}/results", key)
        (dest / "japanfold_manifest.json").write_text(json.dumps(res, indent=2))
    except ApiError:
        pass
    marker.write_text("ok")
    return dest


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_auth(args):
    if args.auth_cmd == "login":
        key = args.api_key or os.environ.get("JAPANFOLD_API_KEY")
        if not key and sys.stdin.isatty():
            key = input("API key (jf_live_...): ").strip()
        if not key:
            _err("no key provided (pass --api-key, set JAPANFOLD_API_KEY, or run interactively)")
            return EXIT_AUTH
        base = resolve_base_url(args)
        try:
            _request("GET", base, "/v1/jobs?limit=1", key)
        except ApiError as e:
            _err(f"key rejected by {base}: {e}")
            return EXIT_AUTH
        cfg = _load_config()
        cfg["api_key"] = key
        cfg["base_url"] = base
        _save_config(cfg)
        print(f"Logged in to {base}. Credentials saved to {CONFIG_PATH}.")
        return 0
    if args.auth_cmd == "status":
        key = resolve_api_key(args)
        base = resolve_base_url(args)
        if not key:
            print(f"Using the free public demo at {base} (no API key). "
                  f"Same limits as the web app.")
            return EXIT_OK
        try:
            _request("GET", base, "/v1/jobs?limit=1", key)
            print(f"Authenticated to {base} (key {key[:12]}…).")
            return EXIT_OK
        except ApiError as e:
            print(f"Key present but not valid at {base}: {e}")
            return EXIT_AUTH
    if args.auth_cmd == "logout":
        cfg = _load_config()
        cfg.pop("api_key", None)
        _save_config(cfg)
        print("Logged out.")
        return EXIT_OK
    return EXIT_INPUT


def cmd_models(args):
    base = resolve_base_url(args)
    _, data, _ = _request("GET", base, "/v1/models", resolve_api_key(args))
    if args.json:
        _emit(data, True)
        return 0
    print("Models:")
    for m in data["models"]:
        print(f"  {m['id']:16} {m.get('tagline', '')}")
    print("Design protocols:")
    for p in data["protocols"]:
        print(f"  {p['id']:24} {p.get('name', '')}")
    print("Embedding models:")
    for m in data.get("embed_models", []):
        print(f"  {m['id']:16} {m.get('tagline', '')}")
    return 0


def cmd_schema(args):
    """Print the OpenAPI 3.1 contract (agent/tool introspection)."""
    base = resolve_base_url(args)
    _, spec, _ = _request("GET", base, "/v1/openapi.json", None)
    print(json.dumps(spec, indent=2))
    return 0


def cmd_predict(args):
    base, key = resolve_base_url(args), resolve_api_key(args)
    fields = _read_input(args.input_pos, args.sequence, args.input)
    if not fields:
        _err("provide an input: a FASTA/YAML file, '-', --sequence, or --input")
        return EXIT_INPUT
    params = _collect_params(args.param, use_msa_server=(True if args.use_msa_server else None),
                             fast=(True if args.fast else None),
                             recycling_steps=args.recycling_steps,
                             sampling_steps=args.sampling_steps,
                             diffusion_samples=args.diffusion_samples,
                             output_format=args.output_format)
    body = {"model": args.model, "name": args.name, "params": params, **fields}
    _, job, _ = _request("POST", base, "/v1/predictions", key, body=body,
                         idempotency=args.idempotency_key)
    return _after_submit(base, key, job, args)


def cmd_design(args):
    base, key = resolve_base_url(args), resolve_api_key(args)
    fields = _read_input(args.input_pos, None, args.spec)
    spec = fields.get("input")
    if not spec:
        _err("provide a design spec: a YAML file, '-', or --spec")
        return EXIT_INPUT
    params = _collect_params(args.param, num_designs=args.num_designs,
                             budget=args.budget, fast=(True if args.fast else None))
    body = {"protocol": args.protocol, "name": args.name, "spec": spec, "params": params}
    _, job, _ = _request("POST", base, "/v1/designs", key, body=body)
    return _after_submit(base, key, job, args)


def cmd_embed(args):
    base, key = resolve_base_url(args), resolve_api_key(args)
    fields = _read_input(args.input_pos, args.sequence, args.input)
    if not fields:
        _err("provide an input: a FASTA/YAML file, '-', --sequence, or --input")
        return EXIT_INPUT
    params = _collect_params(args.param, pool=args.pool, format=args.format,
                             fast=(True if args.fast else None))
    body = {"model": args.model, "name": args.name, "params": params, **fields}
    _, job, _ = _request("POST", base, "/v1/embeddings", key, body=body,
                         idempotency=args.idempotency_key)
    return _after_submit(base, key, job, args)


def _after_submit(base, key, job, args):
    job_id = job["id"]
    _err(f"Submitted job {job_id}  ({job.get('kind')}, {job.get('model') or job.get('protocol')})")
    if not args.wait:
        _emit(job, args.json)
        if not args.json:
            print(job_id)
        return 0
    return _finish(base, key, job_id, args)


def _finish(base, key, job_id, args):
    """Poll a job to completion, download its results, and emit. Shared by
    ``predict/design --wait`` and the standalone ``download`` command."""
    final = _poll_until_done(base, key, job_id, quiet=args.json)
    if final.get("status") != "succeeded":
        _err(f"job {final.get('status')}: {final.get('error') or 'see logs'}")
        _emit(final, args.json)
        raise Done(1)
    dest = _download(base, key, job_id, args.out or ".", name=args.name)
    _err(f"Done. Results in {dest}")
    _emit({**final, "output_dir": str(dest)}, args.json)
    if not args.json:
        print(str(dest))
    return 0


def cmd_jobs(args):
    base, key = resolve_base_url(args), resolve_api_key(args)
    if args.jobs_cmd == "list":
        path = f"/v1/jobs?limit={args.limit}" + (f"&cursor={args.cursor}" if args.cursor else "")
        _, data, _ = _request("GET", base, path, key)
        if args.json:
            _emit(data, True)
            return 0
        for j in data["data"]:
            print(f"  {j['id']}  {j['status']:9} {j.get('kind'):7} {j.get('name', '')}")
        if data.get("has_more"):
            print(f"  … more (next cursor: {data['next_cursor']})", file=sys.stderr)
        return 0
    if args.jobs_cmd == "get":
        _, job, _ = _request("GET", base, f"/v1/jobs/{args.job_id}", key)
        if args.json:
            _emit(job, True)
        else:
            print(f"{job['id']}  {_progress(job)}")
        return 0 if job.get("status") != "failed" else 1
    if args.jobs_cmd == "cancel":
        _, res, _ = _request("POST", base, f"/v1/jobs/{args.job_id}/cancel", key)
        _emit(res, args.json)
        if not args.json:
            print("canceled" if res.get("canceled") else "not canceled")
        return 0
    return EXIT_INPUT


def cmd_download(args):
    base, key = resolve_base_url(args), resolve_api_key(args)
    return _finish(base, key, args.job_id, args)


def cmd_logs(args):
    base, key = resolve_base_url(args), resolve_api_key(args)
    _, blob, _ = _request("GET", base, f"/v1/jobs/{args.job_id}/logs", key, raw=True)
    sys.stdout.write(blob.decode(errors="replace"))
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser():
    # Global flags live on a parent parser (with SUPPRESS defaults) added to the
    # top parser AND every subparser, so `--json`/`--base-url`/`--api-key` work in
    # any position — before OR after the subcommand — without clobbering.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default=argparse.SUPPRESS,
                        help=f"API base URL (default {DEFAULT_BASE_URL}).")
    common.add_argument("--api-key", default=argparse.SUPPRESS, help="Override the API key.")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="Machine-readable JSON on stdout.")

    def add(subparsers, name, **kw):
        return subparsers.add_parser(name, parents=[common], **kw)

    p = argparse.ArgumentParser(prog="japanfold", parents=[common],
                                description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"japanfold {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = add(sub, "auth", help="Log in / check / log out.")
    asub = a.add_subparsers(dest="auth_cmd", required=True)
    add(asub, "login")
    add(asub, "status")
    add(asub, "logout")

    add(sub, "models", help="List available models, protocols and limits.")
    add(sub, "schema", help="Print the OpenAPI 3.1 contract (for tooling/agents).")

    pr = add(sub, "predict", help="Predict a structure (Boltz-2/ESMFold2/Protenix).")
    pr.add_argument("input_pos", nargs="?", metavar="INPUT", help="FASTA/YAML file, or '-' for stdin.")
    pr.add_argument("--sequence", help="A single protein sequence (convenience).")
    pr.add_argument("--input", help="Inline FASTA/YAML string.")
    pr.add_argument("--model", default="boltz2",
                    choices=["boltz2", "esmfold2", "esmfold2-fast", "protenix-v2"])
    pr.add_argument("--name")
    pr.add_argument("--fast", action="store_true")
    pr.add_argument("--use-msa-server", action="store_true", dest="use_msa_server")
    pr.add_argument("--recycling-steps", type=int, dest="recycling_steps")
    pr.add_argument("--sampling-steps", type=int, dest="sampling_steps")
    pr.add_argument("--diffusion-samples", type=int, dest="diffusion_samples")
    pr.add_argument("--output-format", choices=["cif", "pdb"], dest="output_format")
    pr.add_argument("--param", action="append", help="Extra param key=value (repeatable).")
    pr.add_argument("--idempotency-key", dest="idempotency_key")
    pr.add_argument("--wait", action="store_true", help="Poll to completion and download results.")
    pr.add_argument("--out", help="Output directory for --wait (default: cwd).")

    de = add(sub, "design", help="Design binders/proteins (BoltzGen).")
    de.add_argument("input_pos", nargs="?", metavar="SPEC", help="YAML design spec file, or '-'.")
    de.add_argument("--spec", help="Inline YAML design spec.")
    de.add_argument("--protocol", choices=["protein-anything", "peptide-anything", "nanobody-anything",
                    "antibody-anything", "protein-small_molecule", "protein-redesign"])
    de.add_argument("--name")
    de.add_argument("--num-designs", type=int, dest="num_designs")
    de.add_argument("--budget", type=int)
    de.add_argument("--fast", action="store_true")
    de.add_argument("--param", action="append")
    de.add_argument("--wait", action="store_true")
    de.add_argument("--out")

    em = add(sub, "embed", help="Compute ESMC protein-language-model embeddings.")
    em.add_argument("input_pos", nargs="?", metavar="INPUT", help="FASTA/YAML file, or '-' for stdin.")
    em.add_argument("--sequence", help="A single protein sequence (convenience).")
    em.add_argument("--input", help="Inline FASTA/YAML string.")
    em.add_argument("--model", default="esmc-600m", choices=["esmc-300m", "esmc-600m", "esmc-6b"])
    em.add_argument("--pool", default="mean", choices=["mean", "max", "cls"])
    em.add_argument("--format", default="npz", choices=["npz", "parquet"], dest="format")
    em.add_argument("--name")
    em.add_argument("--fast", action="store_true")
    em.add_argument("--param", action="append", help="Extra param key=value (repeatable).")
    em.add_argument("--idempotency-key", dest="idempotency_key")
    em.add_argument("--wait", action="store_true", help="Poll to completion and download results.")
    em.add_argument("--out", help="Output directory for --wait (default: cwd).")

    j = add(sub, "jobs", help="List / inspect / cancel jobs.")
    jsub = j.add_subparsers(dest="jobs_cmd", required=True)
    jl = add(jsub, "list")
    jl.add_argument("--limit", type=int, default=20)
    jl.add_argument("--cursor")
    add(jsub, "get").add_argument("job_id")
    add(jsub, "cancel").add_argument("job_id")

    dl = add(sub, "download", help="Poll to completion and download a job's results.")
    dl.add_argument("job_id")
    dl.add_argument("--out")
    dl.add_argument("--name", help="Subdirectory name (default: job id).")

    lg = add(sub, "logs", help="Print a job's run log.")
    lg.add_argument("job_id")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Materialize SUPPRESS-defaulted globals so handlers can read them freely.
    args.json = getattr(args, "json", False)
    args.base_url = getattr(args, "base_url", None)
    args.api_key = getattr(args, "api_key", None)
    if not hasattr(args, "name"):
        args.name = None
    handlers = {"auth": cmd_auth, "models": cmd_models, "predict": cmd_predict,
                "design": cmd_design, "embed": cmd_embed, "jobs": cmd_jobs,
                "download": cmd_download, "logs": cmd_logs, "schema": cmd_schema}
    try:
        return handlers[args.cmd](args) or EXIT_OK
    except Done as e:
        return e.code
    except ApiError as e:
        _err(str(e))
        return e.exit_code
    except KeyboardInterrupt:
        _err("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
