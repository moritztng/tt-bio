"""Landing-page contact form: deliver visitor messages to the maintainer.

The destination addresses live only in this module (overridable via env) —
nothing served to the browser ever contains them.

Every accepted submission is delivered through THREE independent legs, each
individually failure-isolated, so no submission is ever silently lost:

  1. mail    — first configured provider wins: SMTP (``CONTACT_SMTP_HOST``/
               ``PORT``/``USER``/``PASS``; Gmail is smtp.gmail.com:587 with
               STARTTLS and an app password) or Resend (``RESEND_API_KEY``).
               ``Reply-To`` is the submitter, so answering goes to them.
  2. store   — one JSON line per submission appended to ``CONTACT_LOG``
               (default ``~/japanfold-contact.jsonl``, created 0600).
  3. notify  — a one-line Telegram alert via ``CONTACT_NOTIFY_CMD`` (default
               the japanfold-agent's notify.sh, which holds the bot
               credentials; nothing secret is read or logged here).

A mail-leg failure never fails the request: the message is already safe in
the store and on Telegram, so the visitor gets a success and the failure is
logged server-side with the reason. With no provider configured the endpoint
still works — store + notify only — until a credential lands.

Config comes from the process environment, falling back to a KEY=VALUE file
(``CONTACT_ENV``, default ``.contact.env`` at the repo root — chmod 600,
git-ignored, never committed). Environment wins over the file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import smtplib
import subprocess
import threading
import time
import urllib.request
from collections import deque
from email.message import EmailMessage
from pathlib import Path

from flask import jsonify, request

from .http_common import client_ip

log = logging.getLogger(__name__)

_DEFAULT_TO = "mthuening@tenstorrent.com"
_DEFAULT_CC = "moritz.thuening@gmail.com"
_DEFAULT_NOTIFY_CMD = "/home/cust-team/.japanfold-agent/notify.sh"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_KEYS = (
    "CONTACT_TO", "CONTACT_CC", "CONTACT_FROM",
    "CONTACT_SMTP_HOST", "CONTACT_SMTP_PORT", "CONTACT_SMTP_USER",
    "CONTACT_SMTP_PASS", "CONTACT_SMTP_STARTTLS", "RESEND_API_KEY",
    "CONTACT_LOG", "CONTACT_NOTIFY_CMD",
)

# Field caps (the form mirrors them as maxlength attributes for friendliness).
_MAX = {"name": 120, "email": 200, "company": 160, "message": 5000}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_HEADER_UNSAFE = re.compile(r"[\r\n]+")  # stripped from anything landing in a mail header

# Sliding-window rate limits. Module-level state is per-process; a restart
# resets the budget, which is acceptable for a contact form.
_WINDOW_S = 3600
_PER_IP_PER_WINDOW = 5
_GLOBAL_PER_WINDOW = 50
_lock = threading.Lock()
_hits: dict[str, deque] = {}
_global_hits: deque = deque()


def _rate_limited(ip: str, now: float) -> bool:
    with _lock:
        hits = _hits.setdefault(ip, deque())
        while hits and now - hits[0] > _WINDOW_S:
            hits.popleft()
        while _global_hits and now - _global_hits[0] > _WINDOW_S:
            _global_hits.popleft()
        if len(hits) >= _PER_IP_PER_WINDOW or len(_global_hits) >= _GLOBAL_PER_WINDOW:
            return True
        hits.append(now)
        _global_hits.append(now)
        if len(_hits) > 1000:  # prune idle buckets so the map can't grow forever
            for stale in [k for k, v in _hits.items() if not v]:
                del _hits[stale]
        return False


def _config() -> dict:
    cfg: dict[str, str] = {}
    path = os.environ.get("CONTACT_ENV") or str(_REPO_ROOT / ".contact.env")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    except Exception:
        log.exception("contact: could not read env file %s", path)
    for k in _ENV_KEYS:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    smtp_port = int(cfg.get("CONTACT_SMTP_PORT") or 587)
    return {
        "to": cfg.get("CONTACT_TO") or _DEFAULT_TO,
        "cc": cfg.get("CONTACT_CC") or _DEFAULT_CC,
        "from": cfg.get("CONTACT_FROM") or cfg.get("CONTACT_SMTP_USER")
                or "JapanFold <noreply@japanfold.com>",
        "smtp_host": cfg.get("CONTACT_SMTP_HOST", ""),
        "smtp_port": smtp_port,
        "smtp_user": cfg.get("CONTACT_SMTP_USER", ""),
        "smtp_pass": cfg.get("CONTACT_SMTP_PASS", ""),
        "smtp_starttls": cfg.get("CONTACT_SMTP_STARTTLS", "1").lower() not in ("0", "false", "")
                         and smtp_port != 465,
        "resend_key": cfg.get("RESEND_API_KEY", ""),
        "log": cfg.get("CONTACT_LOG") or str(Path.home() / "japanfold-contact.jsonl"),
        "notify_cmd": cfg.get("CONTACT_NOTIFY_CMD") or _DEFAULT_NOTIFY_CMD,
    }


def _send_smtp(cfg: dict, msg: EmailMessage) -> None:
    cls = smtplib.SMTP_SSL if cfg["smtp_port"] == 465 else smtplib.SMTP
    with cls(cfg["smtp_host"], cfg["smtp_port"], timeout=20) as s:
        if cfg["smtp_starttls"]:
            s.starttls()
        if cfg["smtp_user"]:
            s.login(cfg["smtp_user"], cfg["smtp_pass"])
        s.send_message(msg)


def _send_resend(cfg: dict, msg: EmailMessage) -> None:
    payload = {
        "from": msg["From"], "to": [msg["To"]],
        "reply_to": msg["Reply-To"], "subject": msg["Subject"],
        "text": msg.get_content(),
    }
    if msg["Cc"]:
        payload["cc"] = [msg["Cc"]]
    req = urllib.request.Request(
        "https://api.resend.com/emails", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {cfg['resend_key']}",
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        resp.read()


def _send_mail(cfg: dict, msg: EmailMessage) -> str:
    """One-line outcome for the store record; raises on provider failure."""
    if cfg["smtp_host"]:
        _send_smtp(cfg, msg)
        return "sent:smtp"
    if cfg["resend_key"]:
        _send_resend(cfg, msg)
        return "sent:resend"
    return "skipped:no-provider"


def _store(cfg: dict, record: dict) -> None:
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(cfg["log"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _notify(cfg: dict, record: dict) -> None:
    cmd = cfg["notify_cmd"]
    if not os.access(cmd, os.X_OK):
        log.warning("contact: notify leg skipped, not executable: %s", cmd)
        return
    line = (f"JapanFold contact form: {record['name']} <{record['email']}>"
            + (f" ({record['company']})" if record["company"] else "")
            + f" [{record['ip']}/{record['country'] or '??'}]"
            + f" — {record['message'][:280]}")
    subprocess.run([cmd, line], timeout=15, capture_output=True, check=False)


def _submit(body: dict):
    # Honeypot: a human never sees (or fills) the "website" field. Bots that do
    # get an indistinguishable success and are otherwise dropped entirely.
    if str(body.get("website") or "").strip():
        return jsonify({"ok": True})

    fields = {}
    for name, cap in _MAX.items():
        value = str(body.get(name) or "").strip()
        if len(value) > cap:
            return jsonify({"error": f"{name.capitalize()} is too long (limit {cap:,} characters)."}), 400
        fields[name] = value
    for required in ("name", "email", "message"):
        if not fields[required]:
            return jsonify({"error": f"{required.capitalize()} is required."}), 400
    if not _EMAIL_RE.match(fields["email"]):
        return jsonify({"error": "That doesn't look like a valid email address."}), 400

    ip = client_ip()
    if _rate_limited(ip, time.time()):
        return jsonify({"error": "Too many messages — please try again later."}), 429

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "name": _HEADER_UNSAFE.sub(" ", fields["name"]),
        "email": fields["email"],  # the regex above already excludes CR/LF
        "company": _HEADER_UNSAFE.sub(" ", fields["company"]),
        "message": fields["message"],
        "ip": ip,
        "country": request.headers.get("CF-IPCountry", "").strip(),
    }

    cfg = _config()
    msg = EmailMessage()
    msg["Subject"] = (f"[JapanFold] contact from {record['name']}"
                      + (f" ({record['company']})" if record["company"] else ""))
    msg["From"] = cfg["from"]
    msg["To"] = cfg["to"]
    if cfg["cc"]:
        msg["Cc"] = cfg["cc"]
    msg["Reply-To"] = record["email"]
    msg.set_content(
        f"Name: {record['name']}\n"
        f"Email: {record['email']}\n"
        f"Company: {record['company'] or '-'}\n"
        f"Time: {record['ts']}\n"
        f"IP: {record['ip']} ({record['country'] or '??'})\n"
        f"\n{record['message']}\n")

    try:
        record["mail"] = _send_mail(cfg, msg)
    except Exception as e:
        # The visitor still gets a success: the store + notify legs below carry
        # the message. Log the reason so the mail leg can be fixed.
        log.exception("contact: mail leg failed: %s", e)
        record["mail"] = f"failed:{type(e).__name__}"

    try:
        _store(cfg, record)
    except Exception:
        log.exception("contact: store leg failed")
    try:
        _notify(cfg, record)
    except Exception:
        log.exception("contact: notify leg failed")
    return jsonify({"ok": True})


def register(app) -> None:
    """Register POST /api/contact. Must run BEFORE the SPA catch-all, which
    404s any ``api/`` path it doesn't recognise."""

    @app.post("/api/contact")
    def contact_submit():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        return _submit(body)
