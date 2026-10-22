"""Feature parity of tt-bio's vendored RFD4 prep against upstream, on every shipped example.

Two references, both exact (no tolerance):

* ``prep``: upstream ``rfd4_proteina_prep.prepare_cif`` run from its clone, which is the
  host path NVIDIA's own pipeline calls. Every key of the returned batch must match,
  padding and schema defaults included. An input upstream refuses must be refused with
  the same exception type and message.
* ``golden``: the raw ``feats`` RFD4's own repo produces (``capture_goldens.py``). The
  batch carries them unpadded in the leading slice, and the padded tail must hold the
  schema default.

Each implementation runs in its own processes, so the two atomworks copies never share
an interpreter (atomworks registers annotations globally at import).

    rfd4-ref-env/venv/bin/python -I prep_parity.py [--jobs N] [--seed 0] [--only SUBSTR]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from capture_goldens import BENCH, OUT as GOLDENS, discover  # noqa: E402

PREP_SRC = Path("/home/moritz/rfd4-prep-src/repo/src")
WORK = Path("/home/moritz/.coworker/state/rfd4/parity/io-prep")
REPORT = Path("/home/moritz/.coworker/state/rfd4/parity/io-features.json")
IMPLS = {
    "upstream": ("rfd4_proteina_prep", str(PREP_SRC)),
    "vendored": ("tt_bio._vendor.rfd4_prep", str(REPO)),
}


def to_numpy(t):
    import torch

    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:  # numpy has no bf16: compare the bits
        return t.view(torch.int16).numpy()
    return t.numpy()


def worker(impl: str, seed: int, pairs: list[tuple[str, str]]) -> None:
    import importlib

    import torch

    mod = importlib.import_module(IMPLS[impl][0] + ".processing")
    for src, dst in pairs:
        dst = Path(dst)
        if dst.exists() or dst.with_suffix(".err").exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            feats, _ = mod.prepare_cif(src, seed=seed, device=torch.device("cpu"))
        except Exception as e:  # a refusal is a result, compared like any other
            dst.with_suffix(".err").write_text(f"{type(e).__name__}: {e}")
            continue
        np.savez_compressed(dst, **{k: to_numpy(v[0]) for k, v in feats.items()})


def run_impl(impl: str, seed: int, jobs: list, n: int, chunk: int) -> None:
    pairs = [(src, str(WORK / impl / Path(dst).relative_to(GOLDENS))) for _, src, dst in jobs]
    todo = [p for p in pairs
            if not Path(p[1]).exists() and not Path(p[1]).with_suffix(".err").exists()]
    env = dict(os.environ, PYTHONPATH=IMPLS[impl][1])
    chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]

    def go(c):
        cmd = [sys.executable, "-c",
               f"import sys,json;sys.path.insert(0,{str(HERE)!r});import prep_parity as p;"
               f"p.worker({impl!r},{seed},json.loads(sys.stdin.read()))"]
        r = subprocess.run(cmd, input=json.dumps(c), text=True, env=env, capture_output=True)
        if r.returncode:
            print(f"  {impl} worker died: {r.stderr[-400:]}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(n) as ex:
        for i, _ in enumerate(ex.map(go, chunks), 1):
            print(f"[{impl}] chunk {i}/{len(chunks)} {time.time() - t0:.0f}s", flush=True)


def load(p: Path):
    if p.with_suffix(".err").exists():
        return ("error", p.with_suffix(".err").read_text())
    if not p.exists():
        return ("missing", "")
    with np.load(p) as z:
        return ("ok", {k: z[k] for k in z.files})


def exact(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)


def score_prep(ours, ref) -> list[str]:
    if ours[0] != ref[0]:
        return [f"outcome {ours[0]} vs upstream {ref[0]}: {ours[1] if ours[0] == 'error' else ref[1]}"[:300]]
    if ours[0] == "error":
        return [] if ours[1] == ref[1] else [f"different refusal: {ours[1]!r} vs {ref[1]!r}"[:300]]
    o, r = ours[1], ref[1]
    bad = [f"missing {k}" for k in sorted(set(r) - set(o))]
    bad += [f"extra {k}" for k in sorted(set(o) - set(r))]
    bad += [f"{k} {o[k].shape}/{o[k].dtype} vs {r[k].shape}/{r[k].dtype}"
            if o[k].shape != r[k].shape or o[k].dtype != r[k].dtype else f"{k} values"
            for k in sorted(set(o) & set(r)) if not exact(o[k], r[k])]
    return bad


def score_golden(ours, golden: dict, specs: dict) -> list[str]:
    """The unpadded golden must sit in the batch's leading slice."""
    import torch

    if ours[0] != "ok":
        return [f"vendored {ours[0]}"]
    o, bad = ours[1], []
    for gk, g in golden.items():
        if not gk.startswith("feats/"):
            continue
        k = gk[len("feats/"):]
        if k not in o:
            bad.append(f"missing {k}")
            continue
        v = o[k]
        if v.ndim != g.ndim or any(vs < gs for vs, gs in zip(v.shape, g.shape)):
            bad.append(f"{k} {v.shape} cannot hold golden {g.shape}")
            continue
        head = v[tuple(slice(0, s) for s in g.shape)]
        # prepare_cif casts every field to its schema dtype, as the model receives it; the
        # golden goes through the identical cast. Exact means bit-equal after it, not close.
        want = to_numpy(torch.as_tensor(g, dtype=specs[k][0])) if k in specs else g
        if head.dtype != want.dtype:
            bad.append(f"{k} dtype {head.dtype} vs golden-under-schema {want.dtype}")
        elif not np.array_equal(head, want, equal_nan=True):
            bad.append(f"{k} values")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="substring filter on the input path")
    a = ap.parse_args()

    jobs = [j for j in discover() if a.only in j[1]]
    for impl in IMPLS:
        run_impl(impl, a.seed, jobs, a.jobs, a.chunk)

    # the schema dtypes, from upstream's own definition
    sys.path.insert(0, str(PREP_SRC))
    from rfd4_proteina_prep._cif.datapipes_schema_proteina import SPECS
    rows, fields = [], {}
    for fam, src, dst in jobs:
        rel = Path(dst).relative_to(GOLDENS)
        ours, ref = load(WORK / "vendored" / rel), load(WORK / "upstream" / rel)
        g = load(Path(dst))
        vs_prep = score_prep(ours, ref)
        vs_golden = score_golden(ours, g[1], SPECS) if g[0] == "ok" else []
        if ours[0] == "ok":
            for k in ours[1]:
                fields.setdefault(k, 0)
                fields[k] += any(b.split()[0] in (k, "missing") and k in b for b in vs_prep)
        rows.append({"input": str(Path(src).relative_to(BENCH)), "family": fam, "outcome": ours[0],
                     "vs_prep": vs_prep, "vs_golden": vs_golden})

    prep_ok = sum(not r["vs_prep"] for r in rows)
    gold_rows = [r for r in rows if load(GOLDENS / Path(r["input"]).parent /
                                         (Path(r["input"]).name.split(".cif")[0] + ".npz"))[0] == "ok"]
    gold_ok = sum(not r["vs_golden"] for r in gold_rows)
    report = {
        "seed": a.seed, "prep_pin": "c99a272df90566bbcabf4af0e06bb26e58d704b9",
        "upstream_sha": "2db07bb444d91695f8c10e6acb7648e9297f9221",
        "inputs": len(rows), "refused_identically": sum(r["outcome"] == "error" and not r["vs_prep"] for r in rows),
        "exact_vs_prep": prep_ok, "exact_vs_golden": f"{gold_ok}/{len(gold_rows)}",
        "fields": sorted(fields), "fields_ever_wrong_vs_prep": sorted(k for k, n in fields.items() if n),
        "failures": [r for r in rows if r["vs_prep"] or r["vs_golden"]],
    }
    if not a.only:
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(rows)} inputs: exact vs upstream prep {prep_ok}/{len(rows)} "
          f"({report['refused_identically']} refused identically), exact vs RFD4 goldens {gold_ok}/{len(gold_rows)}")
    for r in report["failures"][:15]:
        print(" ", r["input"], r["vs_prep"][:3], r["vs_golden"][:3])
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
