#!/usr/bin/env python3
"""Run one JapanFold API cell end to end: submit, poll, fetch, check, record.

The correctness sweep's live half. One cell = one submission with an expected
outcome ("ok" or "reject"), and a cell only passes if the service does what the
catalog promises AND -- for an accepted predict/design job -- every structure it
returns survives check_structure.py.

    jf_cell.py --cell single_protein_boltz2 --kind predict --expect ok \\
               --payload payload.json --out results.jsonl

Notes that cost time to rediscover:
  * urllib gets a 403 from the edge; requests/curl with a normal UA do not.
  * The key is optional. Without one the caller is the public tier, scoped to
    its IP, with a per-IP cap of 8 active jobs and 40 submits/min.
  * Artifacts are listed in /v1/jobs/<id>/results, not guessable from the id.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

API = "https://api.japanfold.com"
HERE = Path(__file__).resolve().parent


def call(method: str, path: str, key: str | None = None, body: dict | None = None,
         raw: bool = False, timeout: int = 60):
    cmd = ["curl", "-s", "-m", str(timeout), "-w", "\n%{http_code}", "-X", method, API + path]
    if key:
        cmd += ["-H", f"Authorization: Bearer {key}"]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "--data-binary", json.dumps(body)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    text, _, code = r.stdout.rpartition("\n")
    status = int(code or 0)
    if raw:
        return status, text
    try:
        return status, json.loads(text)
    except ValueError:
        return status, {"raw": text[:400]}


def fetch_artifact(job: str, relpath: str, dest: Path, key: str | None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-s", "-m", "120", "-o", str(dest),
           f"{API}/v1/jobs/{job}/artifacts/{relpath}"]
    if key:
        cmd += ["-H", f"Authorization: Bearer {key}"]
    subprocess.run(cmd, check=False)
    return dest.exists() and dest.stat().st_size > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--kind", required=True, choices=("predict", "design", "embed"))
    ap.add_argument("--expect", default="ok", choices=("ok", "reject"))
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--input", type=Path, help="the target file, for the composition check")
    ap.add_argument("--key")
    ap.add_argument("--out", type=Path, default=Path("results.jsonl"))
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--deadline", type=int, default=2000, help="seconds before giving up")
    a = ap.parse_args()

    payload = json.loads(a.payload.read_text())
    ep = {"predict": "/v1/predictions", "design": "/v1/designs", "embed": "/v1/embeddings"}[a.kind]
    t0 = time.time()
    # 429 on submit is the rate limiter, not an answer about the input: keyless callers get
    # 12 submits/min and 3 active jobs per session. Back off and retry, so a cell records
    # what the service thinks of the payload rather than how fast the runner was going.
    for attempt in range(12):
        status, resp = call("POST", ep, a.key, payload)
        if status != 429:
            break
        time.sleep(10 + 5 * attempt)
    row = {"cell": a.cell, "kind": a.kind, "expect": a.expect, "submit_status": status,
           "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if a.expect == "reject":
        row["detail"] = resp.get("detail", "")
        # A rejection must be a 400 that names the limit, not a 5xx and not a
        # 503 (which only means the pool is down and says nothing about input).
        row["pass"] = status == 400 and bool(row["detail"])
        row["why"] = "" if row["pass"] else f"expected 400 with a message, got {status}"
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(("PASS " if row["pass"] else "FAIL ") + a.cell + f"  [{status}] {row['detail'][:120]}")
        return 0 if row["pass"] else 1

    if status != 202:
        row.update(pass_=False, pass_reason=f"submit returned {status}",
                   detail=resp.get("detail", ""))
        row["pass"] = False
        row["why"] = f"submit returned {status}: {resp.get('detail', '')[:200]}"
        with a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"FAIL {a.cell}  submit {status}: {resp.get('detail','')[:160]}")
        return 1

    job = resp["id"]
    row["job"] = job
    while time.time() - t0 < a.deadline:
        time.sleep(15)
        _, j = call("GET", f"/v1/jobs/{job}", a.key)
        if j.get("status") in ("succeeded", "failed", "cancelled"):
            break
    row["wall_s"] = round(time.time() - t0, 1)
    row["status"] = j.get("status", "timeout")
    row["error"] = j.get("error")
    if row["status"] != "succeeded":
        _, log = call("GET", f"/v1/jobs/{job}/logs", a.key, raw=True)
        row["log_tail"] = "\n".join(log.splitlines()[-12:])
        row["pass"] = False
        row["why"] = f"job {row['status']}: {str(j.get('error'))[:200]}"
        with a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"FAIL {a.cell}  {row['status']} after {row['wall_s']}s\n{row.get('log_tail','')}")
        return 1

    _, res = call("GET", f"/v1/jobs/{job}/results", a.key)
    row["rows"] = res.get("rows")
    checks = []
    for art in res.get("artifacts", []):
        if art.get("type") != "structure":
            continue
        dest = a.artifacts / a.cell / art["path"]
        if not fetch_artifact(job, art["path"], dest, a.key):
            checks.append({"path": art["path"], "verdict": "FAIL", "fail": ["artifact download failed"]})
            continue
        rep = dest.with_suffix(dest.suffix + ".check.json")
        cmd = [sys.executable, str(HERE / "check_structure.py"), str(dest),
               "--kind", "design" if a.kind == "design" else "predict",
               "--json", str(rep), "--quiet"]
        if a.input:
            cmd += ["--input", str(a.input)]
        subprocess.run(cmd, check=False)
        checks.append(json.loads(rep.read_text()) if rep.exists()
                      else {"path": art["path"], "verdict": "FAIL", "fail": ["checker produced nothing"]})
    row["checks"] = [{"path": c.get("struct"), "verdict": c.get("verdict"), "fail": c.get("fail")}
                     for c in checks]
    bad = [c for c in checks if c.get("verdict") == "FAIL"]
    if a.kind != "embed" and not checks:
        row["pass"], row["why"] = False, "job succeeded but returned no structure"
    else:
        row["pass"] = not bad
        row["why"] = "; ".join(f for c in bad for f in c.get("fail", []))[:400]
    with a.out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(("PASS " if row["pass"] else "FAIL ") + f"{a.cell}  {row['wall_s']}s  {row['why']}")
    return 0 if row["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
