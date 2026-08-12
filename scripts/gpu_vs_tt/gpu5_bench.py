#!/usr/bin/env python3
"""boltz-2 / OpenFold3 / ESMFold2 leg of the five-model 512 aa GPU benchmark.

gpu_bench.py already covers the protenix family (protenix-v2, OpenDDE) with the model
loaded once in-process. The other three ship as CLIs or as a plain torch API, so this
script reproduces the same timing scope for them:

  * one process per model, weights loaded ONCE,
  * the model's own predict step wrapped in torch.cuda.synchronize() on both sides,
  * fold 1 discarded as cold, then N warm folds, reported as median with n/min/max,
  * a kernel counter that proves cuEquivariance actually ran instead of trusting the flag.

The two CLI models get four copies of the same target under four different names, so one
process does cold + 3 warm with no reload. Emits the same JSON shape gpu_bench.py does, so
both legs land in one table.

Usage:
    /root/venv-boltz/bin/python3 gpu5_bench.py --model boltz-2 --repeat 3 \
        --yaml perf/size512/fixtures/cdk2x2_512.yaml --a3m .../cdk2x2_512.a3m \
        --out /root/results/gpu_boltz-2_prot512_h200.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SEED = 0
SAMPLES = 1
# Recycles and sampling steps are each model's own shipped default, matching what the TT
# arm runs per model (tt_bio/main.py::_resolve_recycling_steps / _resolve_sampling_steps).
# Forcing one value across five models would detune the ones that ship lower, and ESMFold2
# clips 200 back to 68 anyway, so "200 everywhere" is not even reachable.
DEFAULTS = {
    "boltz-2":   dict(recycles=3,  steps=200),
    "openfold3": dict(recycles=3,  steps=200),
    "esmfold2":  dict(recycles=10, steps=100),   # 100 requested -> 68 executed
}


# --------------------------------------------------------------------------------------
# kernel counters
# --------------------------------------------------------------------------------------
def install_cueq_counters() -> dict:
    """Wrap every triangle-* entry point in cuequivariance_ops_torch with a counter.

    Patching the vendor package rather than each model's call site is what makes this
    evidence: a rung that *asks* for a kernel is not proof the kernel *ran*. Both
    OpenFold3's docs and protenix's own code fall back to torch on unsupported shapes,
    silently. If the counter is 0 after a fold, cuEquivariance did not run, whatever the
    flag said.
    """
    counts: dict[str, int] = {}
    try:
        mod = importlib.import_module("cuequivariance_ops_torch")
    except ImportError:
        counts["cuequivariance_ops_torch"] = -1   # -1 == package not importable
        return counts
    for name in dir(mod):
        if "triangle" not in name.lower():
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        counts[name] = 0

        def make(nm, orig):
            def wrapper(*a, **kw):
                counts[nm] += 1
                return orig(*a, **kw)
            return wrapper
        try:
            setattr(mod, name, make(name, fn))
        except Exception:
            counts.pop(name, None)
    return counts


def install_sdpa_counter() -> dict:
    """Count torch SDPA calls. Not a pass/fail gate, just the other half of the picture:
    a model with 0 cueq calls and thousands of SDPA calls is running the torch path."""
    counts = {"torch_sdpa": 0}
    torch = importlib.import_module("torch")
    orig = torch.nn.functional.scaled_dot_product_attention

    def wrapper(*a, **kw):
        counts["torch_sdpa"] += 1
        return orig(*a, **kw)
    torch.nn.functional.scaled_dot_product_attention = wrapper
    return counts


# --------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------
class FoldTimer:
    """Wraps one method so each call is timed with a CUDA sync on both sides.

    The sync matters more than it looks: an unsynced region lets the next host call
    absorb device time and has inverted a ranking before now.
    """

    def __init__(self):
        self.times: list[float] = []

    def patch(self, obj, attr: str):
        torch = importlib.import_module("torch")
        orig = getattr(obj, attr)

        def wrapper(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            torch.cuda.synchronize()
            self.times.append(time.perf_counter() - t0)
            return out
        setattr(obj, attr, wrapper)
        return orig


def summarize(times: list[float], repeat: int) -> dict:
    """Fold 1 is the cold fold and is discarded explicitly."""
    if not times:
        return dict(error="no folds were timed")
    cold, warm = times[0], times[1:]
    if not warm:
        return dict(cold_s=round(cold, 3), error="no warm folds")
    ts = sorted(warm)
    return dict(
        cold_s=round(cold, 3),
        warm_times_s=[round(t, 3) for t in warm],
        warm_n=len(warm),
        warm_min_s=round(ts[0], 3),
        warm_median_s=round(ts[len(ts) // 2], 3),
        warm_max_s=round(ts[-1], 3),
        warm_spread_pct=round(100.0 * (ts[-1] - ts[0]) / ts[len(ts) // 2], 2),
    )


# --------------------------------------------------------------------------------------
# boltz-2
# --------------------------------------------------------------------------------------
def run_boltz(args) -> dict:
    """boltz 2.2.1. cuEquivariance kernels are ON by default (--no_kernels defaults
    False and DISABLES them), so the fast path needs no flag -- only the [cuda] extra.
    --use_potentials defaults False and stays off."""
    torch = importlib.import_module("torch")
    seq = args.seq_file.read_text().strip()
    work = Path(args.work) / "boltz"
    inp = work / "input"
    shutil.rmtree(work, ignore_errors=True)
    inp.mkdir(parents=True)

    # Four copies of the same target under four names: one process, one weight load,
    # cold + 3 warm. The MSA is the pinned a3m -- adding the `msa:` key is the ONLY
    # change against the committed fixture YAML.
    base = args.yaml.read_text()
    assert "msa:" not in base, "fixture YAML already carries an msa key"
    a3m = str(args.a3m.resolve())
    n = args.repeat + 1
    for i in range(n):
        y = base.replace(f"sequence: {seq}", f"sequence: {seq}\n      msa: {a3m}")
        assert "msa:" in y, "failed to inject the msa key -- sequence line not found"
        (inp / f"fold{i}.yaml").write_text(y)

    cueq = install_cueq_counters()
    sdpa = install_sdpa_counter()
    timer = FoldTimer()

    # Time the model's own predict step, not the CLI: featurization, weight load and
    # mmCIF writing all sit outside it, which is the same scope gpu_bench.py and the TT
    # harness use.
    mod = importlib.import_module("boltz.model.models.boltz2")
    cls = getattr(mod, "Boltz2")
    assert hasattr(cls, "predict_step"), f"no predict_step on {cls}"
    timer.patch(cls, "predict_step")

    from boltz.main import cli
    argv = ["predict", str(inp), "--out_dir", str(work / "out"),
            "--cache", os.environ.get("BOLTZ_CACHE", "/root/.boltz"),
            "--devices", "1", "--accelerator", "gpu",
            "--recycling_steps", str(args.recycles),
            "--sampling_steps", str(args.steps),
            "--diffusion_samples", str(SAMPLES),
            "--output_format", "mmcif", "--num_workers", "2"]
    if args.extra:
        argv += args.extra.split()
    print("boltz argv:", " ".join(argv), file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    try:
        cli.main(args=argv, standalone_mode=False)
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    wall = time.perf_counter() - t0

    out = summarize(timer.times, args.repeat)
    preds = sorted((work / "out").rglob("*.cif"))
    return dict(out, wall_s=round(wall, 2), n_timed_calls=len(timer.times),
                kernel_counts_total=dict(cueq, **sdpa),
                predictions=[str(p) for p in preds],
                cli_argv=argv, msa_rows=args.a3m.read_text().count(">"),
                n_residues=len(seq))


# --------------------------------------------------------------------------------------
# OpenFold3
# --------------------------------------------------------------------------------------
def run_of3(args) -> dict:
    """openfold3 0.4.4. Two shipped defaults must be overridden and both are traps:
    --num-diffusion-samples defaults to 5 (set 1, one structure is what a researcher asks
    for and what every other model here does) and --use-msa-server defaults True, which
    would silently ignore the precomputed MSA and hit ColabFold. cuEquivariance is opt-in
    through the runner YAML."""
    seq = args.seq_file.read_text().strip()
    work = Path(args.work) / "of3"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    a3m = str(args.a3m.resolve())

    n = args.repeat + 1
    queries = [dict(name=f"fold{i}", chains=[dict(
        molecule_type="protein", chain_ids=["A"], sequence=seq,
        main_msa_file_paths=[a3m])]) for i in range(n)]
    qpath = work / "queries.json"
    qpath.write_text(json.dumps(queries, indent=2))

    ypath = work / "runner.yaml"
    ypath.write_text(
        "model_update:\n  presets: [\"predict\"]\n  custom:\n    settings:\n"
        "      memory:\n        eval:\n          use_cueq_triangle_kernels: true\n")

    cueq = install_cueq_counters()
    sdpa = install_sdpa_counter()
    timer = FoldTimer()
    patched = None
    for path, attr in (("openfold3.model.openfold3", "OpenFold3"),
                       ("openfold3.projects.of3_all_atom.model", "OpenFold3")):
        try:
            cls = getattr(importlib.import_module(path), attr)
        except Exception:
            continue
        for m in ("predict_step", "forward"):
            if hasattr(cls, m):
                timer.patch(cls, m)
                patched = f"{path}.{attr}.{m}"
                break
        if patched:
            break
    assert patched, "could not find OpenFold3's predict entry point -- introspect on the box"
    print("of3 timing:", patched, file=sys.stderr, flush=True)

    argv = ["--queries", str(qpath), "--output-dir", str(work / "out"),
            "--checkpoint-path", args.checkpoint,
            "--runner-config", str(ypath),
            "--num-diffusion-samples", str(SAMPLES),
            "--no-use-msa-server", "--seed", str(SEED)]
    if args.extra:
        argv += args.extra.split()
    t0 = time.perf_counter()
    entry = importlib.import_module("openfold3.cli.predict")
    fn = getattr(entry, "main", None) or getattr(entry, "cli")
    try:
        fn(args=argv, standalone_mode=False) if hasattr(fn, "main") else fn(argv)
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    wall = time.perf_counter() - t0

    out = summarize(timer.times, args.repeat)
    preds = sorted((work / "out").rglob("*.cif"))
    return dict(out, wall_s=round(wall, 2), n_timed_calls=len(timer.times),
                timed_symbol=patched, kernel_counts_total=dict(cueq, **sdpa),
                predictions=[str(p) for p in preds], cli_argv=argv,
                msa_rows=args.a3m.read_text().count(">"), n_residues=len(seq))


# --------------------------------------------------------------------------------------
# ESMFold2
# --------------------------------------------------------------------------------------
def run_esmfold2(args) -> dict:
    """ESMFold2 is single-sequence by design, so the MSA is dropped -- a property of the
    model, not of the harness. Plain torch API, so the loop needs no CLI wrapper."""
    torch = importlib.import_module("torch")
    seq = args.seq_file.read_text().strip()
    work = Path(args.work) / "esmfold2"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    cueq = install_cueq_counters()
    sdpa = install_sdpa_counter()

    load_t0 = time.perf_counter()
    mod = importlib.import_module(args.esm_module)
    ESMFold2Model = getattr(mod, "ESMFold2Model")
    model = ESMFold2Model.from_pretrained(args.esm_repo).cuda().eval()
    load_s = time.perf_counter() - load_t0

    times, preds = [], []
    fold = getattr(model, "fold", None) or getattr(model, "infer", None)
    assert fold is not None, f"no fold/infer method on {ESMFold2Model}"
    kw = dict(num_recycles=args.recycles, num_steps=args.steps)
    for i in range(args.repeat + 1):
        torch.manual_seed(SEED)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = fold(seq, **kw)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        p = work / f"fold{i}.pdb"
        text = out if isinstance(out, str) else getattr(out, "pdb", None)
        if text:
            p.write_text(text if isinstance(text, str) else text[0])
            preds.append(str(p))

    return dict(summarize(times, args.repeat), load_s=round(load_s, 2),
                kernel_counts_total=dict(cueq, **sdpa), predictions=preds,
                fold_kwargs=kw, msa_rows=0,
                msa_note="single-sequence model: the 35-row MSA is not consumed",
                n_residues=len(seq))


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["boltz-2", "openfold3", "esmfold2"])
    ap.add_argument("--repeat", type=int, default=3, help="warm folds; fold 1 is cold on top")
    ap.add_argument("--yaml", type=Path, default=HERE.parents[1] /
                    "perf/size512/fixtures/cdk2x2_512.yaml")
    ap.add_argument("--a3m", type=Path, default=HERE.parents[1] /
                    "perf/size512/fixtures/cdk2x2_512.a3m")
    ap.add_argument("--seq-file", type=Path, default=HERE / "fixtures/prot512.seq")
    ap.add_argument("--checkpoint", default="/root/ckpt/of3-p2-155k.pt")
    ap.add_argument("--esm-module", default="esm.models.esmfold2")
    ap.add_argument("--esm-repo", default="biohub/ESMFold2")
    ap.add_argument("--recycles", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--work", default="/root/work")
    ap.add_argument("--extra", default=None, help="extra CLI args, verbatim")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = DEFAULTS[args.model]
    if args.recycles is None:
        args.recycles = d["recycles"]
    if args.steps is None:
        args.steps = d["steps"]

    torch = importlib.import_module("torch")
    seq = args.seq_file.read_text().strip()
    # The a3m's query row must be the sequence, or the two sides are not folding the
    # same target. Same identical-bytes check gpu_bench.py makes.
    if args.model != "esmfold2":
        rows = args.a3m.read_text().split("\n")
        assert rows[1] == seq, f"{args.a3m} query row does not match {args.seq_file}"

    fn = {"boltz-2": run_boltz, "openfold3": run_of3, "esmfold2": run_esmfold2}[args.model]
    t0 = time.perf_counter()
    try:
        res = fn(args)
        err = None
    except Exception as e:
        import traceback
        res, err = {}, traceback.format_exc()
        print(err, file=sys.stderr)

    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    pkgs = {}
    for p in ("torch", "boltz", "openfold3", "esm", "cuequivariance_torch",
              "cuequivariance_ops_torch", "transformers"):
        try:
            pkgs[p] = importlib.metadata.version(p)
        except Exception:
            pass

    summary = dict(
        model=args.model, side="gpu", gpu=torch.cuda.get_device_name(0),
        gpu_capability=list(torch.cuda.get_device_capability()),
        host_cpu=cpu, cpu_count=os.cpu_count(),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        recycling_steps=args.recycles, sampling_steps=args.steps,
        diffusion_samples=SAMPLES, seed=SEED,
        fixture=dict(yaml=str(args.yaml), a3m=str(args.a3m), n_residues=len(seq)),
        packages=pkgs, session_wall_s=round(time.perf_counter() - t0, 1),
        error=err, result=res, date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in (res or {}).items()
                      if k not in ("predictions", "cli_argv")}, indent=2))
    return 1 if err else 0


if __name__ == "__main__":
    import importlib.metadata  # noqa: F401  (populated lazily above)
    sys.exit(main())
