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
                       {"model": "boltzgen", "protocol": "binder", "spec": boltzgen}, "ok"))
    if rfd3:
        design.append(("rfd3", "designs",
                       {"model": "rfd3", "protocol": "rfd3", "structure": rfd3[0],
                        "contig": rfd3[1], "params": {"num_designs": 1, "steps": 20}}, "ok"))
    return design + [
        ("boltz2-trpcage", "predictions", {"model": "boltz2", "sequence": TRPCAGE}, "ok"),
        ("boltz2-640", "predictions", {"model": "boltz2", "sequence": seq640}, "ok"),
        ("boltz2-1024", "predictions", {"model": "boltz2", "sequence": seq1024}, "ok"),
        ("esmfold2-fast-trpcage", "predictions",
         {"model": "esmfold2-fast", "sequence": TRPCAGE}, "ok"),
        # Expected to fail: out of DRAM at every size with fast mode on. Recorded, not hidden.
        ("esmfold2-trpcage", "predictions", {"model": "esmfold2", "sequence": TRPCAGE}, "fail"),
        ("protenix-v2-trpcage", "predictions",
         {"model": "protenix-v2", "sequence": TRPCAGE}, "ok"),
        ("protenix-v2-640", "predictions", {"model": "protenix-v2", "sequence": seq640}, "ok"),
        ("opendde-trpcage", "predictions", {"model": "opendde", "sequence": TRPCAGE}, "ok"),
        ("esmc-300m", "embeddings", {"model": "esmc-300m", "sequence": TRPCAGE}, "ok"),
        ("esmc-600m", "embeddings", {"model": "esmc-600m", "sequence": TRPCAGE}, "ok"),
        ("esmc-6b", "embeddings", {"model": "esmc-6b", "sequence": TRPCAGE}, "ok"),
        ("saprot-650m", "embeddings", {"model": "saprot-650m", "sequence": TRPCAGE}, "ok"),
        ("saprot-1.3b", "embeddings", {"model": "saprot-1.3b", "sequence": TRPCAGE}, "ok"),
    ]


def call(url: str, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://api.japanfold.com")
    ap.add_argument("--out", required=True)
    ap.add_argument("--poll-timeout", type=int, default=2400)
    ap.add_argument("--seq640", required=True, help="a 640 aa sequence (K3's band)")
    ap.add_argument("--seq1024", required=True, help="a 1024 aa sequence (the catalog cap)")
    ap.add_argument("--boltzgen-spec", help="path to a BoltzGen YAML design spec")
    ap.add_argument("--rfd3-structure", help="path to a PDB/mmCIF target for RFD3")
    ap.add_argument("--rfd3-contig", default="A1-10,20,A31-40")
    a = ap.parse_args()

    bg = open(a.boltzgen_spec).read() if a.boltzgen_spec else None
    rf = (open(a.rfd3_structure).read(), a.rfd3_contig) if a.rfd3_structure else None
    if not (bg and rf):
        print("NOTE: design rows skipped -- pass --boltzgen-spec and --rfd3-structure to "
              "cover all twelve served models.")

    submitted = []
    for label, ep, body, expect in rows(a.seq640, a.seq1024, rfd3=rf, boltzgen=bg):
        code, resp = call(f"{a.base}/v1/{ep}", body)
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
            state = (j.get("status") or j.get("state") or "").lower()
            r["state"], r["progress"] = state, j.get("progress")
            if state in ("succeeded", "done", "completed", "failed", "error", "cancelled"):
                r["terminal"] = state
                r["detail"] = j.get("error") or j.get("message")
                print(f"  {label:24s} -> {state}")
                del pending[label]
    for r in pending.values():
        r["terminal"] = "TIMEOUT"

    bad = []
    for r in submitted:
        ok = r.get("terminal") in ("succeeded", "done", "completed")
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
