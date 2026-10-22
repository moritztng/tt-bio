"""Did lifting OpenFold3's fold out of the serving method change what the service folds?

`_predict_openfold3_one` now calls `worker.openfold3_host_inputs` and `openfold3_device_fold`
instead of carrying both inline (commits 3d9d63cdf, 0e4309239). The move was verbatim and the host
half is covered on CPU, but the device half has not folded since, and nothing in it is believable
until it has.

The OF3 accuracy gate does NOT cover this. `tests/test_openfold3_fold_rmsd.py` builds its own
features and calls `OpenFold3.fold` directly, so it passes whether or not the serving path works.
The only thing that exercises the extracted callables is a `tt-bio predict` through the CLI, which
is what this script runs.

    # on the card, in this worktree
    python3 perf/rfd4/verify_of3_refactor.py fold --card 0 --out after.json

    # and in a detached worktree at the commit before the move, for the baseline
    git worktree add --detach /tmp/of3-premove 3d9d63cdf~1
    python3 perf/rfd4/verify_of3_refactor.py fold --card 0 --tree /tmp/of3-premove --out before.json

    python3 perf/rfd4/verify_of3_refactor.py compare before.json after.json

Same seed, same fixture, same flags: the metrics row should match to the printed tolerance. The
confidence numbers are the sharp ones -- a featurization that lost a feature moves pLDDT and pTM
long before it moves anything a smoke test looks at.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Small, single-sequence, no MSA fetch: the point is the code path, not the biology.
FIXTURE = ROOT / "perf" / "size512" / "fixtures" / "cdk2x2_256.yaml"
#: Keys worth comparing. plddt/ptm/iptm move on any featurization change; n_tokens and
#: msa_depth catch a query that was built differently.
COMPARED = ("plddt", "ptm", "iptm", "confidence_score", "n_tokens", "n_atoms", "msa_depth",
            "n_residues", "n_chains")


def fold(args) -> int:
    tree = Path(args.tree).resolve() if args.tree else ROOT
    out_dir = Path(args.work) / ("premove" if args.tree else "head")
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(args.fixture),
           "--model", "openfold3", "--out_dir", str(out_dir), "--seed", "0",
           "--single_sequence", "--sampling_steps", str(args.sampling_steps),
           "--diffusion_samples", "1"]
    if args.recycling_steps is not None:
        cmd += ["--recycling_steps", str(args.recycling_steps)]
    env = {**os.environ, "TT_VISIBLE_DEVICES": args.card}
    print(" ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=tree, env=env, timeout=args.timeout).returncode
    if rc != 0:
        print(f"fold failed rc={rc}")
        return rc

    sys.path.insert(0, str(tree))
    from tt_bio.main import predict_results_dir_name

    results = out_dir / predict_results_dir_name("openfold3", Path(args.fixture).stem) / "results.json"
    rows = json.loads(results.read_text())
    row = next(r for r in rows if r.get("status") == "ok")
    record = {"tree": str(tree), "commit": _commit(tree), "fixture": str(args.fixture),
              "card": args.card, "row": {k: row.get(k) for k in COMPARED},
              "runtime_s": row.get("runtime_s")}
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


def _commit(tree: Path) -> str:
    return subprocess.run(["git", "-C", str(tree), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def compare_rows(before: dict, after: dict, tol: float = 1e-6) -> list[str]:
    """Keys that moved. Empty means the refactor changed nothing the service reports."""
    drift = []
    for key in COMPARED:
        a, b = before.get(key), after.get(key)
        if a is None and b is None:
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) > tol:
                drift.append(f"{key}: {a} -> {b}")
        elif a != b:
            drift.append(f"{key}: {a!r} -> {b!r}")
    return drift


def compare(args) -> int:
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    drift = compare_rows(before["row"], after["row"], tol=args.tol)
    print(f"before {before['commit']}  after {after['commit']}  tol {args.tol}")
    if not drift:
        print("IDENTICAL -- the extraction changed nothing the service reports")
        return 0
    print("DRIFT, the extraction is not verbatim after all:")
    for line in drift:
        print(f"  {line}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fold", help="fold the fixture in a tree and record its metrics row")
    f.add_argument("--card", required=True, help="the leased card, as TT_VISIBLE_DEVICES")
    f.add_argument("--tree", default=None, help="a checkout to fold in; default is this one")
    f.add_argument("--fixture", default=str(FIXTURE))
    f.add_argument("--sampling_steps", type=int, default=20)
    f.add_argument("--recycling_steps", type=int, default=None)
    f.add_argument("--timeout", type=int, default=3600)
    f.add_argument("--work", default=str(ROOT / "perf" / "rfd4" / "work" / "verify"))
    f.add_argument("--out", required=True)
    f.set_defaults(func=fold)

    c = sub.add_parser("compare", help="diff two recorded rows")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--tol", type=float, default=1e-6)
    c.set_defaults(func=compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
