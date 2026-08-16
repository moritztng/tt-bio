#!/usr/bin/env python3
"""Multi-shape in one submission, arm 2: ten shapes at once, against ten shapes alone.

Arm 1 (`multishape_arm1.sh`) is the decisive form and has no card to run on -- every chip on
UF-EV-A13-GWH02 belongs to the JapanFold pool or the customer (state doc 12.4). This is what is
left, and it measures what a user actually gets rather than what one process does:

  batch  one POST /v1/predictions with 10 targets at 10 different sizes (`max_complexes` is 10)
  solo   the same 10 targets, one submission each

It does NOT isolate a process -- targets fan across workers, so two results are two devices and
bit-exactness is not the bar. The bar is:

  1. every target comes back;
  2. **the right target's sequence is in the right file** -- the failure to hunt is one target's
     output appearing under another target's name, which is what shape-keyed state corruption
     looks like from outside (`tt-bio-trace-replay-shape-keyed-multitarget-corruption`);
  3. every structure passes check_structure.py, in both arrangements;
  4. batch and solo agree on length per target.

The ten sizes are all different on purpose: a swap between two targets of the same length would
be invisible to a length check, and every one of these is distinguishable by residue count alone.

    multishape_arm2.py [--model esmfold2-fast] [--out results/multishape_arm2.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from matrix import cdk2  # noqa: E402  the same fixture construction the size axis uses

API = "https://api.japanfold.com"
SIZES = [64, 128, 192, 256, 320, 384, 448, 512, 640, 1024]


def call(method: str, path: str, body: dict | None = None, timeout: int = 90):
    cmd = ["curl", "-s", "-m", str(timeout), "-w", "\n%{http_code}", "-X", method, API + path]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "--data-binary", json.dumps(body)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    text, _, code = r.stdout.rpartition("\n")
    try:
        return int(code or 0), json.loads(text)
    except ValueError:
        return int(code or 0), {"raw": text[:400]}


def wait(job: str, deadline: int) -> dict:
    t0 = time.time()
    j = {}
    while time.time() - t0 < deadline:
        time.sleep(15)
        _, j = call("GET", f"/v1/jobs/{job}")
        if j.get("status") in ("succeeded", "failed", "cancelled"):
            break
    return j


def submit(model: str, targets: list[dict], name: str, deadline: int) -> tuple[str, dict]:
    for attempt in range(12):
        st, resp = call("POST", "/v1/predictions",
                        {"model": model, "name": name, "targets": targets})
        if st != 429:
            break
        time.sleep(10 + 5 * attempt)
    if st != 202:
        return "", {"status": f"submit {st}", "detail": resp.get("detail", "")}
    return resp["id"], wait(resp["id"], deadline)


def fetch_and_check(job: str, art_dir: Path, inputs: dict[str, str]) -> dict:
    """Download every structure and check it against the YAML of the target it is filed under."""
    _, res = call("GET", f"/v1/jobs/{job}/results")
    out = {}
    for art in res.get("artifacts", []):
        if art.get("type") != "structure":
            continue
        tid = art.get("target") or art["path"].split(".")[0]
        dest = art_dir / art["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-s", "-m", "180", "-o", str(dest),
                        f"{API}/v1/jobs/{job}/artifacts/{art['path']}"], check=False)
        yml = art_dir / f"{tid}.input.yaml"
        yml.write_text(inputs[tid])
        rep = dest.with_suffix(dest.suffix + ".check.json")
        subprocess.run([sys.executable, str(HERE / "check_structure.py"), str(dest),
                        "--input", str(yml), "--json", str(rep), "--quiet"], check=False)
        r = json.loads(rep.read_text()) if rep.exists() else {"verdict": "FAIL",
                                                              "fail": ["checker produced nothing"]}
        n_res = sum(c["n_res"] for c in r.get("checks", {}).get("chains", []))
        out[tid] = {"verdict": r.get("verdict"), "fail": r.get("fail", []), "n_res": n_res}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2-fast",
                    help="single-sequence, and the only model measured to fold every one of "
                         "these sizes on this box (size axis, 7/7)")
    ap.add_argument("--deadline", type=int, default=2400)
    ap.add_argument("--artifacts", type=Path, default=HERE / "results" / "artifacts")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "multishape_arm2.json")
    a = ap.parse_args()

    yamls = {n: f"sequences:\n  - protein: {{id: A, sequence: {cdk2(n)}}}\n" for n in SIZES}
    targets = [{"content": yamls[n], "name": f"cdk2_{n}"} for n in SIZES]
    rep: dict = {"model": a.model, "sizes": SIZES, "fail": [], "warn": []}

    print(f"batch: one submission, {len(SIZES)} targets at {SIZES}")
    job, j = submit(a.model, targets, "multishape_arm2_batch", a.deadline)
    rep["batch"] = {"job": job, "status": j.get("status"), "error": j.get("error")}
    if j.get("status") != "succeeded":
        rep["fail"].append(f"batch job {j.get('status')}: {str(j.get('error'))[:200]}")

    # Each artifact comes back filed under the NAME the submission gave its target
    # (`cdk2_640`), not a positional `target_N`, and the artifacts arrive in arbitrary
    # order. So the check is name -> size -> the YAML for that size: if the service ever
    # files one target's output under another's name, the residue count disagrees with the
    # name, and no two of the ten sizes are equal so no swap can hide.
    _, res = call("GET", f"/v1/jobs/{job}/results") if job else (0, {})
    rep["batch"]["rows"] = {r["id"]: r.get("n_residues") for r in res.get("rows", [])}
    inputs = {}
    for art in res.get("artifacts", []):
        tid = art.get("target") or ""
        n = int(tid.rsplit("_", 1)[-1]) if tid.rsplit("_", 1)[-1].isdigit() else None
        inputs[tid] = yamls.get(n, "")
    batch = fetch_and_check(job, a.artifacts / "multishape_arm2_batch", inputs) if job else {}
    rep["batch"]["per_target"] = batch

    for tid, r in sorted(batch.items()):
        want = int(tid.rsplit("_", 1)[-1]) if tid.rsplit("_", 1)[-1].isdigit() else None
        if want not in SIZES:
            rep["fail"].append(f"batch returned an artifact named {tid!r}, which is not one "
                               f"of the ten targets that were submitted")
            continue
        if r["n_res"] != want:
            rep["fail"].append(f"batch {tid} is {r['n_res']} residues, not {want} -- a "
                               f"target's output is filed under another target's name")
        if r["verdict"] == "FAIL":
            rep["fail"].append(f"batch {tid}: " + "; ".join(r["fail"])[:200])
    missing = [n for n in SIZES if f"cdk2_{n}" not in batch]
    if missing:
        rep["fail"].append(f"batch returned {len(batch)} structures for {len(SIZES)} targets; "
                           f"missing {missing}")

    print(f"solo: {len(SIZES)} submissions, one target each")
    rep["solo"] = {}
    for n in SIZES:
        sjob, sj = submit(a.model, [{"content": yamls[n], "name": f"cdk2_{n}"}],
                          f"multishape_arm2_solo_{n}", a.deadline)
        one = (fetch_and_check(sjob, a.artifacts / f"multishape_arm2_solo_{n}",
                               {f"cdk2_{n}": yamls[n]}) if sjob and sj.get("status") == "succeeded"
               else {})
        got = one.get(f"cdk2_{n}", {})
        rep["solo"][n] = {"job": sjob, "status": sj.get("status"),
                          "verdict": got.get("verdict"), "n_res": got.get("n_res"),
                          "fail": got.get("fail", [])}
        if sj.get("status") != "succeeded":
            rep["fail"].append(f"solo {n} aa: job {sj.get('status')}")
        elif got.get("verdict") == "FAIL":
            rep["fail"].append(f"solo {n} aa: " + "; ".join(got.get("fail", []))[:200])
        elif got.get("n_res") != n:
            rep["fail"].append(f"solo {n} aa returned {got.get('n_res')} residues")
        print(f"  solo {n:5d} aa  {sj.get('status')}  {got.get('verdict')}  n_res={got.get('n_res')}")

    # Same target, two arrangements: the lengths must agree. They are two devices, so the
    # coordinates need not, and this does not claim they do.
    for n, s in rep["solo"].items():
        b = batch.get(f"cdk2_{n}", {})
        if b.get("n_res") and s.get("n_res") and b["n_res"] != s["n_res"]:
            rep["fail"].append(f"{n} aa: batch returned {b['n_res']} residues, solo {s['n_res']}")

    rep["verdict"] = "FAIL" if rep["fail"] else "PASS"
    a.out.write_text(json.dumps(rep, indent=1))
    print(f"\n{rep['verdict']}")
    for f in rep["fail"]:
        print("  FAIL " + f)
    return 1 if rep["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
