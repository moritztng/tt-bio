#!/usr/bin/env python3
"""§7.5: verify the live service from outside, one real job per served model.

`/api/health` returning 200 is never evidence the service works -- during the
device-safe-restart failure mode it reads 200 while every job sits at progress 0.0
forever. Only a job polled to a terminal state is evidence, so that is what this does.

    python3 perf/whcut/e2e_verify.py --base https://api.japanfold.com --out results.json

Submits against the public v1 API (`POST /v1/predictions|designs|embeddings`), polls
`GET /v1/jobs/<id>`, and records every job id. Exits non-zero if any row's outcome does
not match what §7.5 expects -- including `esmfold2`, which is expected to FAIL: it is a
pre-existing defect (sweep finding 1) and a deploy that reports clean without it is
reporting a fiction.
"""
import argparse
import json
import time
import urllib.error
import urllib.request

TRPCAGE = "NLYIQWLKDGGPSSGRPPPS"
UA = "japanfold-e2e-verify/1.0"

# Rows: (label, endpoint, body, expect) where expect is "ok" or "fail".
# The three extra Boltz-2 sizes are the ones the assembled branch actually moves: 640 is
# where K3 fires, 1024 is lever C's range and the cap the catalog advertises.
def rows(seq640: str, seq1024: str, rfd3=None, boltzgen=None) -> list:
    design = []
    if boltzgen:
        # BoltzGen takes a YAML design spec; RFD3 takes pasted structure text plus a
        # contig, not a spec. Both fixtures are supplied by the caller rather than
        # inlined, so this file cannot drift from whatever the deploy pass actually folds.
        design.append(("boltzgen", "designs",
                       {"model": "boltzgen", "protocol": "protein-anything", "spec": boltzgen}, "ok"))
    if rfd3:
        design.append(("rfd3", "designs",
                       {"model": "rfd3", "protocol": "rfd3-scaffold", "structure": rfd3[0],
                        "contig": rfd3[1], "params": {"num_designs": 1, "steps": 20}}, "ok"))
    big = []
    if seq640:
        big.append(("boltz2-640", "predictions", {"model": "boltz2", "sequence": seq640}, "ok"))
        big.append(("protenix-v2-640", "predictions",
                    {"model": "protenix-v2", "sequence": seq640}, "ok"))
    if seq1024:
        big.append(("boltz2-1024", "predictions", {"model": "boltz2", "sequence": seq1024}, "ok"))
    if seq640:
        # The esmfold2 defect row. The sweep's ladder runs 128 to 1024 aa and OOMs at all
        # seven points (§0.6), so a size in that band is what tests it. Trp-cage at 20 aa is
        # BELOW the whole ladder and folds fine, measured against the live service.
        big.append(("esmfold2-640", "predictions",
                    {"model": "esmfold2", "sequence": seq640}, "fail"))
    return design + big + [
        ("boltz2-trpcage", "predictions", {"model": "boltz2", "sequence": TRPCAGE}, "ok"),
        ("esmfold2-fast-trpcage", "predictions",
         {"model": "esmfold2-fast", "sequence": TRPCAGE}, "ok"),
        # MEASURED succeeding on the live service pre-deploy, job 2abf12b4bff8dd20974abeda1ecf33a8.
        # 20 aa sits below the sweep's 128-1024 aa OOM ladder, so this row shows the model is
        # served, not that the defect is gone. `esmfold2-640` above is the row that tests it.
        ("esmfold2-trpcage", "predictions", {"model": "esmfold2", "sequence": TRPCAGE}, "ok"),
        ("protenix-v2-trpcage", "predictions",
         {"model": "protenix-v2", "sequence": TRPCAGE}, "ok"),
        ("opendde-trpcage", "predictions", {"model": "opendde", "sequence": TRPCAGE}, "ok"),
        ("esmc-300m", "embeddings", {"model": "esmc-300m", "sequence": TRPCAGE}, "ok"),
        ("esmc-600m", "embeddings", {"model": "esmc-600m", "sequence": TRPCAGE}, "ok"),
        ("esmc-6b", "embeddings", {"model": "esmc-6b", "sequence": TRPCAGE}, "ok"),
        ("saprot-650m", "embeddings", {"model": "saprot-650m", "sequence": TRPCAGE}, "ok"),
        ("saprot-1.3b", "embeddings", {"model": "saprot-1.3b", "sequence": TRPCAGE}, "ok"),
    ]


def call(url: str, body=None, timeout=60):
    """Returns (status, parsed). A non-JSON body comes back as {"raw": text} rather than
    raising: an error page or an empty body is exactly what this script exists to report,
    and a JSONDecodeError here would abort the run before a single job was submitted."""
    data = json.dumps(body).encode() if body is not None else None
    # A real User-Agent is REQUIRED, not cosmetic. The API is behind Cloudflare and urllib's
    # default `Python-urllib/3.x` is banned by browser signature: every request comes back
    # 403 with the body `error code: 1010`. Measured this pass -- Python-urllib 403,
    # python-requests 202, curl 202 -- so it is that one token and not Python as such.
    headers = {"User-Agent": UA}
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)

    def parse(text):
        try:
            return json.loads(text or "{}")
        except ValueError:
            return {"raw": text[:400]}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, parse(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, parse(e.read().decode(errors="replace"))
    except urllib.error.URLError as e:
        return 0, {"raw": f"URLError: {e.reason}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://api.japanfold.com")
    ap.add_argument("--out", required=True)
    ap.add_argument("--poll-timeout", type=int, default=2400)
    ap.add_argument("--seq640", help="a 640 aa sequence (K3's band). Omit for a smoke run.")
    ap.add_argument("--seq1024", help="a 1024 aa sequence (the catalog cap). Omit for a smoke run.")
    ap.add_argument("--boltzgen-spec", help="path to a BoltzGen YAML design spec")
    ap.add_argument("--rfd3-structure", help="path to a PDB/mmCIF target for RFD3")
    ap.add_argument("--rfd3-contig", default="A1-10,20,A31-40")
    ap.add_argument("--submit-backoff", type=int, default=20,
                    help="seconds to wait after a 429, multiplied by the attempt number")
    # NOTE for anyone reusing this against the deploy: poll each job BY ID, as this script
    # does. `GET /v1/jobs` is scoped and does not list anonymous submissions -- measured
    # returning 6 jobs, newest from 2026-08-15, while two freshly-submitted folds were
    # running. A drain or completion check built on that list reports idle while work is in
    # flight. The service-wide signal that works is /api/cluster's runs.running.
    a = ap.parse_args()

    bg = open(a.boltzgen_spec).read() if a.boltzgen_spec else None
    rf = (open(a.rfd3_structure).read(), a.rfd3_contig) if a.rfd3_structure else None
    if not (bg and rf):
        print("NOTE: design rows skipped -- pass --boltzgen-spec and --rfd3-structure to "
              "cover all twelve served models.")

    submitted = []
    for label, ep, body, expect in rows(a.seq640, a.seq1024, rfd3=rf, boltzgen=bg):
        # The public demo rate-limits, and this submits a dozen jobs back to back. A 429 is
        # the service working as intended, so back off and retry rather than recording it as
        # a model failure.
        for attempt in range(6):
            code, resp = call(f"{a.base}/v1/{ep}", body)
            if code != 429:
                break
            wait = a.submit_backoff * (attempt + 1)
            print(f"  {label}: 429, backing off {wait}s")
            time.sleep(wait)
        jid = resp.get("id") or resp.get("job_id")
        submitted.append({"label": label, "expect": expect, "submit_code": code,
                          "job_id": jid, "submit_resp": resp if not jid else None})
        print(f"submit {label:24s} HTTP {code} job {jid}")

    deadline = time.time() + a.poll_timeout
    pending = {r["label"]: r for r in submitted if r["job_id"]}
    while pending and time.time() < deadline:
        time.sleep(20)
        for label, r in list(pending.items()):
            code, j = call(f"{a.base}/v1/jobs/{r['job_id']}")
            # The job object carries `done` (0/1) and `error`, not a status string. Polling for
            # `status`/`state` here matched nothing and timed out every row -- checked against a
            # real /v1/jobs body: {created_at, done, error, finished_at, id, kind, links, model,
            # name, object, params, ...}.
            r["progress"] = j.get("progress")
            if j.get("done"):
                r["terminal"] = "failed" if j.get("error") else "succeeded"
                r["detail"] = j.get("error")
                r["finished_at"] = j.get("finished_at")
                print(f"  {label:24s} -> {r['terminal']}"
                      f"{': ' + str(j['error'])[:80] if j.get('error') else ''}")
                del pending[label]
    for r in pending.values():
        r["terminal"] = "TIMEOUT"

    bad = []
    for r in submitted:
        ok = r.get("terminal") == "succeeded"
        if r["expect"] == "ok" and not ok:
            bad.append(f"{r['label']}: expected success, got {r.get('terminal')}")
        if r["expect"] == "fail" and ok:
            bad.append(f"{r['label']}: expected the known pre-existing failure, it SUCCEEDED "
                       "-- good news, but §7.5's expectation is stale and must be updated")
    json.dump({"base": a.base, "rows": submitted, "mismatches": bad}, open(a.out, "w"), indent=1)
    print("\n" + ("MISMATCHES:\n  " + "\n  ".join(bad) if bad else
                  "every row matched its §7.5 expectation"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
