"""API-key store for the JapanFold ``/v1`` customer API.

Keys look like ``jf_live_<43 url-safe chars>``. Only the SHA-256 of each key is
ever persisted, so a leaked store reveals no usable keys. A key maps to a
*customer id*, which the ``/v1`` layer passes to :class:`JobManager` as the
``owner`` — giving per-customer job isolation and quotas for free (the engine
already keys ownership, listing, and throttling purely off that opaque string).

Keys are read from two sources, merged (env wins on hash collision):

1. ``$JAPANFOLD_API_KEYS`` — comma-separated ``customer:key`` pairs, hashed at
   load time. A convenience for dev / ephemeral deploys; never write real keys
   into an env var in production.
2. A JSON file at ``$JAPANFOLD_API_KEYS_FILE`` (default
   ``~/.aiand-bio/api_keys.json``)::

       {"<sha256hex>": {"customer": "acme", "name": "prod key",
                         "created_at": 1782990000.0}}

The file is the source of truth in production; mint keys with
``tt-bio apikey create --customer <id>`` (or ``python -m tt_bio.platform.apikeys``).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

KEY_PREFIX = "jf_live_"
# 32 bytes -> 43 url-safe chars; with the prefix a key is 51 chars.
_SECRET_BYTES = 32

_lock = threading.Lock()
_cache: dict[str, Any] = {"path": None, "mtime": None, "store": None}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def default_store_path() -> Path:
    """Where the key store lives unless overridden by ``$JAPANFOLD_API_KEYS_FILE``."""
    env = os.environ.get("JAPANFOLD_API_KEYS_FILE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".aiand-bio" / "api_keys.json"


def _hash(key: str) -> str:
    return hashlib.sha256(key.strip().encode()).hexdigest()


def generate_key() -> str:
    """A fresh, unguessable key. Shown to the customer exactly once."""
    return KEY_PREFIX + secrets.token_urlsafe(_SECRET_BYTES)


def _valid_format(key: str) -> bool:
    return isinstance(key, str) and key.startswith(KEY_PREFIX) and len(key) >= len(KEY_PREFIX) + 20


def _load_file(path: Path) -> dict[str, dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict) and v.get("customer")}


def _load_env() -> dict[str, dict]:
    raw = os.environ.get("JAPANFOLD_API_KEYS", "").strip()
    if not raw:
        return {}
    out: dict[str, dict] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        customer, key = pair.split(":", 1)
        customer, key = customer.strip(), key.strip()
        if customer and key:
            out[_hash(key)] = {"customer": customer, "name": "env", "created_at": None}
    return out


def _file_store(path: Path) -> dict[str, dict]:
    """The cached ``{sha256: record}`` from the key file, refreshed only when the
    file's mtime changes. Returned by reference (hot path — no per-call copy);
    callers must treat it as read-only."""
    with _lock:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None
        if not (_cache["path"] == str(path) and _cache["mtime"] == mtime and _cache["store"] is not None):
            _cache.update(path=str(path), mtime=mtime, store=_load_file(path))
        return _cache["store"]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def verify(key: str | None, path: Path | None = None) -> str | None:
    """Return the customer id for a key, or ``None`` if unknown/malformed.

    Hot path (every authenticated request): one mtime stat + one dict lookup
    against the cached file store, no copy. The env overlay is consulted only on
    a file miss — and ``_load_env`` returns instantly when ``$JAPANFOLD_API_KEYS``
    is unset (the production case), so env keys are never re-hashed per request.
    """
    if not _valid_format(key or ""):
        return None
    h = _hash(key)
    rec = _file_store(path or default_store_path()).get(h) or _load_env().get(h)
    return rec.get("customer") if rec else None


def mint(customer: str, name: str | None = None, path: Path | None = None) -> tuple[str, dict]:
    """Create a key for ``customer``, persist only its hash, return (key, record).

    The plaintext key is returned once and never stored — surface it to the
    customer immediately.
    """
    customer = str(customer).strip()
    if not customer:
        raise ValueError("customer id is required")
    path = path or default_store_path()
    key = generate_key()
    record = {"customer": customer, "name": name or "", "created_at": time.time()}
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        data[_hash(key)] = record
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.chmod(0o600)
        os.replace(tmp, path)
        _cache.update(path=None, mtime=None, store=None)  # invalidate
    return key, record


def list_keys(path: Path | None = None) -> list[dict]:
    """Non-secret records (hash prefix + customer + name + created_at)."""
    out = []
    for h, rec in _load_file(path or default_store_path()).items():
        out.append({"hash_prefix": h[:12], "customer": rec.get("customer"),
                    "name": rec.get("name", ""), "created_at": rec.get("created_at")})
    out.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return out


def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="tt_bio.platform.apikeys", description="Manage JapanFold API keys.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="Mint a key for a customer.")
    c.add_argument("--customer", required=True)
    c.add_argument("--name", default=None)
    sub.add_parser("list", help="List keys (no secrets).")
    args = p.parse_args(argv)

    if args.cmd == "create":
        key, rec = mint(args.customer, args.name)
        print(f"customer : {rec['customer']}")
        print(f"key      : {key}")
        print("\nStore this key now — it is not recoverable. Give it to the customer as:")
        print("  export JAPANFOLD_API_KEY=" + key)
        return 0
    if args.cmd == "list":
        rows = list_keys()
        if not rows:
            print("(no keys)")
        for r in rows:
            print(f"{r['hash_prefix']}…  {r['customer']:20}  {r['name']}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
