#!/usr/bin/env python3
"""Does the same fold request twice, on the deployed surface, return the same numbers?

This is the property the scoring product is sold on, so it is measured against the served API
rather than a local device. It submits one two-chain complex N times, downloads every structure,
and compares coordinates, the pLDDT B-factor column and each confidence scalar across runs.

    python scripts/dev_api_determinism.py --runs 2 --fast off
    python scripts/dev_api_determinism.py --runs 2 --fast on

`--fast` is the control that matters. Measured 2026-09-24 against
https://dev-api-japanfold.aiand.com (platform d6cf0d222, boltz2, 91 residues, 2 samples):

    coordinates: identical in every run, fast on and off
    confidence : NOT reproducible. Fast off, the 7 runs whose results survive land on exactly
                 two answers, 3 on one and 4 on the other: ipTM 0.878067 vs 0.877723 (3.4e-4),
                 complex_ipde 0.906688 vs 0.909500 (2.8e-3). Fast on, one pair also differed.
                 One further fold failed outright.

Two discrete answers, not a spread, point at a discrete cause: which of the pool's four chips
ran the job, or a cold versus warm path. The dev API and the agent's logs record neither, so
this is unattributed. A first fast-off pair agreed exactly and looked like a clean control; it
had simply landed twice on the same answer. Run a determinism check many times.

So the structure module is reproducible and the nondeterminism is in the confidence head. That
matters here because ipSAE is computed entirely from the confidence head's PAE output: a moving confidence head moves the score even when the structure
does not move at all.

The API exposes no `seed` and no `write_pae`, so this cannot vary the seed and cannot fetch a PAE
matrix to score. Both are noted in state/cmp-score.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

API = "https://dev-api-japanfold.aiand.com"
BINDER = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG"
PARTNER = "GSHMKVLGKIIEYAKQNNVSVNELAKELGVSRQTIYNWLNG"
SCALARS = ("confidence_score", "iptm", "ptm", "complex_plddt", "complex_ipde", "complex_pde")


def curl(*args: str) -> str:
    return subprocess.run(["curl", "-s", "-m", "60", *args], check=True,
                          capture_output=True, text=True).stdout


def submit(fast: bool, samples: int) -> str:
    body = {"model": "boltz2", "name": "cmp-score-determinism",
            "params": {"diffusion_samples": samples, "fast": fast},
            "targets": [{"name": "probe",
                         "content": f">A|protein\n{BINDER}\n>B|protein\n{PARTNER}\n"}]}
    out = json.loads(curl("-X", "POST", f"{API}/v1/predictions",
                          "-H", "Content-Type: application/json", "-d", json.dumps(body)))
    if "id" not in out:
        raise SystemExit(f"submit refused: {out}")
    return out["id"]


def wait(job: str, timeout_s: int = 1200) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        row = json.loads(curl(f"{API}/v1/jobs/{job}"))
        if row["status"] in ("succeeded", "failed", "canceled"):
            return row
        time.sleep(15)
    raise SystemExit(f"job {job} did not finish in {timeout_s}s")


def atoms(path: Path):
    xyz, bfac, ident = [], [], []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            f = line.split()
            xyz.append((float(f[10]), float(f[11]), float(f[12])))
            bfac.append(float(f[17]))
            ident.append((f[5], f[6], f[3], f[15]))
    return xyz, bfac, ident


def fetch(job: str, into: Path) -> list[Path]:
    into.mkdir(parents=True, exist_ok=True)
    res = json.loads(curl(f"{API}/v1/jobs/{job}/results"))
    (into / "results.json").write_text(json.dumps(res))
    out = []
    for art in res.get("artifacts", []):
        p = into / art["path"]
        p.write_text(curl(f"{API}{art['url']}"))
        out.append(p)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--fast", choices=("on", "off"), default="off")
    ap.add_argument("--workdir", default="/tmp/cmpdev-determinism")
    a = ap.parse_args()
    fast = a.fast == "on"
    root = Path(a.workdir) / a.fast
    runs, failed = [], 0
    for i in range(a.runs):
        job = submit(fast, a.samples)
        row = wait(job)
        print(f"run {i}: job {job} {row['status']} (fast={row['params'].get('fast')})")
        if row["status"] != "succeeded":
            failed += 1          # a failed fold is a finding of its own, not a reason to stop
            continue
        runs.append(fetch(job, root / f"run{i}"))
    if len(runs) < 2:
        raise SystemExit(f"{len(runs)} run(s) succeeded, {failed} failed: nothing to compare")

    ref, worst_xyz, worst_b = runs[0], 0.0, 0.0
    for other in runs[1:]:
        for p, q in zip(ref, other):
            if p.name == "results.json":
                continue
            x1, b1, i1 = atoms(p)
            x2, b2, i2 = atoms(q)
            if i1 != i2:
                raise SystemExit(f"atom order differs between runs for {p.name}")
            worst_xyz = max(worst_xyz, max(abs(c - d) for u, v in zip(x1, x2) for c, d in zip(u, v)))
            worst_b = max(worst_b, max(abs(c - d) for c, d in zip(b1, b2)))
    rows = [json.loads((r[0].parent / "results.json").read_text())["rows"][0] for r in runs]
    vectors = {tuple(round(r.get(k, 0.0), 6) for k in SCALARS) for r in rows}
    print(f"\nfast={a.fast}  runs={a.runs}  succeeded={len(runs)}  failed={failed}")
    print(f"  distinct confidence vectors: {len(vectors)} of {len(rows)}")
    print(f"  coordinates : max |dx| {worst_xyz:.6f} A  -> {'IDENTICAL' if worst_xyz == 0 else 'DIFFER'}")
    print(f"  pLDDT column: max |d|  {worst_b:.4f}      -> {'IDENTICAL' if worst_b == 0 else 'DIFFER'}")
    for k in SCALARS:
        vals = [r[k] for r in rows if k in r]
        d = max(vals) - min(vals) if vals else 0.0
        print(f"  {k:17} spread {d:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
