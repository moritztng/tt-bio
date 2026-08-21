"""Run a tt_bio command with the softmax site census installed, then merge the per-process results.

    python3 perf/xmsoftmax/softmax_site_census.py --out results/census_of3_256.json -- \
        predict perf/size512/fixtures/cdk2x2_256.yaml --model openfold3 --single_sequence

Everything after `--` is the tt_bio CLI command, verbatim. The child processes are what actually
touch the device, so the census is installed via PYTHONPATH + sitecustomize rather than by
patching in this process (see sm_census/sitecustomize.py).

PYTHONPATH also pins the repo root FIRST so every process imports tt_bio from THIS checkout. The
venv has tt-bio installed from /home/ttuser/tt-bio-dev, and without the pin the spawned workers
import that tree instead -- which would score a lever that is not in the code being measured
(parity-gate-scores-installed-package-not-checkout).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOOK = os.path.join(ROOT, "perf", "xmsoftmax", "sm_census")


def merge(tmp: str) -> dict:
    merged: dict[str, dict] = {}
    procs = []
    for f in sorted(glob.glob(os.path.join(tmp, "census_pid*.json"))):
        d = json.load(open(f))
        procs.append({"pid": d["pid"], "tt_bio": d["tt_bio"], "n_sites": len(d["sites"])})
        for s in d["sites"]:
            key = f'{s["site"]}|{s["op"]}|{s["shape"]}|{s["dtype"]}'
            m = merged.get(key)
            if m is None:
                merged[key] = dict(s)
                continue
            # Weighted mean over processes; extrema are extrema.
            a, b = m["n_measured"], s["n_measured"]
            if a + b:
                ma = m["rowsum_mean"] or 0.0
                mb = s["rowsum_mean"] or 0.0
                m["rowsum_mean"] = (ma * a + mb * b) / (a + b)
                m["deficit"] = 1.0 - m["rowsum_mean"]
            m["n_calls"] += s["n_calls"]
            m["n_measured"] = a + b
            for k, fn in (("rowsum_min", min), ("rowsum_max", max), ("rowsum_p01_min", min)):
                vals = [v for v in (m[k], s[k]) if v is not None]
                m[k] = fn(vals) if vals else None
            m["errors"] = (m["errors"] + s["errors"])[:3]
    rows = sorted(merged.values(),
                  key=lambda r: (r["deficit"] is None, -(r["deficit"] or 0.0)))
    return {"procs": procs, "sites": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-measured", type=int, default=24)
    ap.add_argument("--python", default="/home/ttuser/tt-bio-dev/env/bin/python")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no tt_bio CLI command after --")

    tmp = tempfile.mkdtemp(prefix="smcensus_")
    env = dict(os.environ)
    env["TT_BIO_SM_CENSUS_DIR"] = tmp
    env["TT_BIO_SM_CENSUS_ROOT"] = ROOT
    env["TT_BIO_SM_CENSUS_MAX"] = str(a.max_measured)
    env["PYTHONPATH"] = os.pathsep.join(
        [ROOT, HOOK] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        rc = subprocess.call([a.python, "-m", "tt_bio.main"] + cmd, env=env, cwd=ROOT)
        res = merge(tmp)
        plog = os.path.join(tmp, "procs.log")
        res["hooked_procs"] = (open(plog).read().splitlines()
                              if os.path.exists(plog) else [])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    res["meta"] = {"cmd": cmd, "rc": rc, "max_measured": a.max_measured, "root": ROOT}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[census] rc={rc}  {len(res['sites'])} site/shape combos, "
          f"{len(res['procs'])} process(es) -> {a.out}")
    for s in res["sites"][:12]:
        print(f"  deficit={s['deficit']!s:>10.10}  min={s['rowsum_min']!s:>10.10}  "
              f"n={s['n_calls']:<6} {s['dtype']:<16} {str(s['shape']):<22} {s['site']} [{s['op']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
