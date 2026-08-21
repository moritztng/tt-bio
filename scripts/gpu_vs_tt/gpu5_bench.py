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
import re
import shutil
import sys
import time
import types
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
# Both cuEquivariance distributions, and the submodules models actually import from.
# OpenFold3 takes triangle_multiplicative_update from cuequivariance_torch and
# triangle_attention from cuequivariance_ops_torch.triangle_attention /
# cuequivariance_torch.primitives.triangle, so wrapping only the top level of
# cuequivariance_ops_torch counts none of it and reads as a silent 0.
_CUEQ_MODULES = (
    "cuequivariance_ops_torch",
    "cuequivariance_ops_torch.triangle_attention",
    "cuequivariance_torch",
    "cuequivariance_torch.primitives.triangle",
)


def install_cueq_counters() -> dict:
    """Wrap every triangle-* entry point in both cuEquivariance packages with a counter.

    Patching the vendor package rather than each model's call site is what makes this
    evidence: a rung that *asks* for a kernel is not proof the kernel *ran*. Both
    OpenFold3's docs and protenix's own code fall back to torch on unsupported shapes,
    silently. If the counter is 0 after a fold, cuEquivariance did not run, whatever the
    flag said.

    MUST be called before the model package is imported. Models bind these functions with
    `from X import Y`, which copies the reference into the model's own namespace; patching
    the source module afterwards leaves that copy pointing at the original and the counter
    reads 0 for a kernel that ran on every call.
    """
    counts: dict[str, int] = {}
    seen = False
    for modname in _CUEQ_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        seen = True
        short = modname.split(".")[-1]
        for name in dir(mod):
            if "triangle" not in name.lower():
                continue
            fn = getattr(mod, name, None)
            if not callable(fn) or isinstance(fn, types.ModuleType):
                continue
            key = name if modname == "cuequivariance_ops_torch" else f"{short}.{name}"
            counts[key] = 0

            def make(nm, orig):
                def wrapper(*a, **kw):
                    counts[nm] += 1
                    return orig(*a, **kw)
                return wrapper
            try:
                setattr(mod, name, make(key, fn))
            except Exception:
                counts.pop(key, None)
    if not seen:
        counts["cuequivariance"] = -1   # -1 == neither package importable
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
        self.gaps: list[float] = []       # host time between one fold and the next
        self._last_end: float | None = None

    def patch(self, obj, attr: str):
        torch = importlib.import_module("torch")
        orig = getattr(obj, attr)

        def wrapper(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if self._last_end is not None:
                self.gaps.append(t0 - self._last_end)
            out = orig(*a, **kw)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.times.append(t1 - t0)
            self._last_end = t1
            return out
        setattr(obj, attr, wrapper)
        return orig


def summarize(times: list[float], repeat: int, gaps: list[float] | None = None) -> dict:
    """Fold 1 is the cold fold and is discarded explicitly.

    ``gaps`` is everything the framework does per target BETWEEN two timed folds --
    featurization of the next one, structure writeout of the last one, framework
    overhead. The published cell is ``warm_median_s``, which excludes all of it; the TT
    side's cell is a whole fold and includes it. So ``warm_gap_median_s`` is this side's
    per-fold host share, and it is the number the two sides differ by. Note it is a
    steady-state gap: where the loader prefetches (boltz-2 runs --num_workers 2), host
    work that overlaps device work does not appear here, and that is correct -- it costs
    the fold nothing.
    """
    if not times:
        return dict(error="no folds were timed")
    cold, warm = times[0], times[1:]
    if not warm:
        return dict(cold_s=round(cold, 3), error="no warm folds")
    ts = sorted(warm)
    out = dict(
        cold_s=round(cold, 3),
        warm_times_s=[round(t, 3) for t in warm],
        warm_n=len(warm),
        warm_min_s=round(ts[0], 3),
        warm_median_s=round(ts[len(ts) // 2], 3),
        warm_max_s=round(ts[-1], 3),
        warm_spread_pct=round(100.0 * (ts[-1] - ts[0]) / ts[len(ts) // 2], 2),
    )
    if gaps:
        wg = sorted(gaps[1:]) or sorted(gaps)   # drop the cold fold's trailing gap
        out.update(warm_gaps_s=[round(g, 3) for g in gaps],
                   warm_gap_median_s=round(wg[len(wg) // 2], 3))
    return out


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

    out = summarize(timer.times, args.repeat, timer.gaps)
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

    # OF3 filters MSA files by BASENAME, not by the path you hand it. parse_msas_direct
    # keys its output dict on the file stem and skips any stem missing from
    # MSASettings.max_seq_counts (io/sequence/msa.py:275), whose default keys are the
    # pipeline's own DB names: uniref90_hits, uniprot_hits, bfd_uniclust_hits,
    # mgnify_hits, colabfold_main and so on. Handing it cdk2x2_512.a3m therefore parses
    # NOTHING, and the empty dict surfaces far downstream as
    # `sorted(all_msas_per_chain.keys())[0] -> IndexError` inside featurization, with the
    # run still exiting 0. Stage a copy named uniref90_hits.a3m: that is the first entry
    # in aln_order and the canonical slot for a precomputed unpaired main MSA, and its
    # 10000-row cap keeps all 35 rows. Bytes are unchanged -- this is a rename, not a
    # conversion, and it is the only per-model input change OF3 needs.
    msa_dir = work / "msa"
    msa_dir.mkdir()
    staged = msa_dir / "uniref90_hits.a3m"
    shutil.copyfile(args.a3m, staged)
    a3m = str(staged.resolve())

    # InferenceQuerySet: seeds at the top level, queries keyed by name. Four names so one
    # process does cold + 3 warm with no weight reload. use_paired_msas False because the
    # pinned fixture is a single unpaired a3m -- leaving it True makes OF3 look for a
    # paired alignment that does not exist.
    n = args.repeat + 1
    qset = dict(seeds=[SEED], queries={
        f"fold{i}": dict(
            chains=[dict(molecule_type="protein", chain_ids=["A"], sequence=seq,
                         main_msa_file_paths=[a3m])],
            use_msas=True, use_paired_msas=False, use_main_msas=True)
        for i in range(n)})
    qpath = work / "queries.json"
    qpath.write_text(json.dumps(qset, indent=2))

    # cuEquivariance is opt-in for OF3, through the runner yaml rather than a flag.
    #
    # The seed has to be set here too. InferenceQuerySet.seeds exists and is parsed, but
    # the dataset is built from experiment_settings.seeds instead
    # (experiment_runner.py:557 and :730), which defaults to [42] -- so a query file
    # asking for seed 0 is silently folded at 42.
    ypath = work / "runner.yaml"
    ypath.write_text(
        f"experiment_settings:\n  seeds: [{SEED}]\n"
        "model_update:\n"
        "  presets: [\"predict\"]\n"
        "  custom:\n"
        "    settings:\n"
        "      memory:\n"
        "        eval:\n"
        "          use_cueq_triangle_kernels: true\n")

    cueq = install_cueq_counters()
    sdpa = install_sdpa_counter()
    timer = FoldTimer()
    runner_mod = importlib.import_module("openfold3.projects.of3_all_atom.runner")
    cls = getattr(runner_mod, "OpenFold3AllAtom")
    assert hasattr(cls, "predict_step"), "no predict_step on OpenFold3AllAtom"
    timer.patch(cls, "predict_step")
    patched = "openfold3.projects.of3_all_atom.runner.OpenFold3AllAtom.predict_step"
    print("of3 timing:", patched, file=sys.stderr, flush=True)

    # --num-diffusion-samples defaults to 5 and --use-msa-server defaults True; the second
    # would silently ignore the precomputed MSA and hit ColabFold, so both are overridden.
    #
    # --num-model-seeds is deliberately NOT passed. Any truthy value makes the runner
    # discard the query set's seeds and substitute generate_seeds(42, n)
    # (experiment_runner.py:610-612), and there is no --start-seed to steer it: passing
    # "1" silently folded at seed 2746317213 instead of the seed 0 every other model here
    # uses, which is exactly the cross-model hyperparameter mismatch rule 4 forbids.
    # Omitting it leaves self.seeds at the query set's own [SEED].
    argv = ["predict", "--query-json", str(qpath), "--output-dir", str(work / "out"),
            "--inference-ckpt-path", args.checkpoint, "--runner-yaml", str(ypath),
            "--num-diffusion-samples", str(SAMPLES),
            "--use-msa-server", "false", "--use-templates", "false"]
    if args.extra:
        argv += args.extra.split()
    print("of3 argv:", " ".join(argv), file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    cli_mod = importlib.import_module("openfold3.run_openfold")
    group = None
    for name in dir(cli_mod):
        obj = getattr(cli_mod, name)
        if hasattr(obj, "main") and hasattr(obj, "commands"):
            group = obj
            break
    assert group is not None, "could not find run_openfold's click group"
    try:
        group.main(args=argv, standalone_mode=False)
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    wall = time.perf_counter() - t0

    out = summarize(timer.times, args.repeat, timer.gaps)
    preds = sorted((work / "out").rglob("*.cif"))
    # OF3 catches a per-fold featurization failure, logs it and carries on, so the CLI
    # exits 0 having predicted nothing. That is how the empty-MSA bug read as rc=0 with
    # timed calls attached. A cell with no structure is a failure, not a fast fold.
    assert preds, ("openfold3 produced no structure: every fold failed inside the runner. "
                   "Read the log for the per-fold traceback -- exit code 0 is not evidence.")
    # OF3 writes the seed it actually used into the output path. Read it back rather than
    # trusting the query set, since the runner is willing to override it.
    seeds_used = sorted({m.group(1) for p in preds
                         if (m := re.search(r"seed_(\d+)", str(p)))})
    assert seeds_used == [str(SEED)], (
        f"openfold3 folded at seed(s) {seeds_used}, not {SEED}: the runner overrode the "
        "query set's seed, so this cell is not comparable to the other four")
    return dict(out, wall_s=round(wall, 2), n_timed_calls=len(timer.times),
                timed_symbol=patched, kernel_counts_total=dict(cueq, **sdpa),
                predictions=[str(p) for p in preds], cli_argv=argv,
                msa_staged_as=str(staged), msa_rows=args.a3m.read_text().count(">"),
                n_residues=len(seq))


# --------------------------------------------------------------------------------------
# ESMFold2
# --------------------------------------------------------------------------------------
def run_esmfold2(args) -> dict:
    """ESMFold2 is single-sequence by design, so the MSA is dropped -- a property of the
    model, not of the harness. Plain torch API, so the loop needs no CLI wrapper.

    The model class ships in transformers, not in the `esm` package and not in the HF
    repo: `biohub/ESMFold2` holds only config.json + model.safetensors + ccd.pkl, so
    trust_remote_code has nothing to fetch, and upstream esm 3.3.0's
    esm.models.esmfold2 carries only the input builders. transformers 4.57.6 defines
    ESMFold2Model in transformers.models.esmfold2.modeling_esmfold2 but its lazy
    __init__ re-exports only ESMFold2ExperimentalModel, so AutoModel cannot resolve it
    and the import has to name the module directly.
    """
    torch = importlib.import_module("torch")
    # torch picks the cuDNN SDPA backend first where it thinks it applies, and on a
    # driver 580 / torch 2.13 H200 it builds no plan for ESMFold2's attention shapes:
    # "cuDNN Frontend error: No valid execution plans built", raised from inside
    # F.scaled_dot_product_attention. Turning that one backend off leaves flash and
    # mem-efficient, which is where this attention wants to run anyway. Recorded in the
    # result so the number is never read as if the default backend produced it.
    cudnn_sdp_disabled = False
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
        cudnn_sdp_disabled = True
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

    # set_kernel_backend picks the fast path: None is the reference implementation,
    # "fused" the vendored Triton kernels, "cuequivariance" the cueq kernels with a
    # python fallback where they do not apply. The default is the reference path, so
    # rule 1 (run it the fast way a researcher would) means selecting one explicitly.
    backend = None if args.esm_backend == "reference" else args.esm_backend
    if hasattr(model, "set_kernel_backend"):
        model.set_kernel_backend(backend)

    times, gaps, preds = [], [], []
    # forward's own knob names: num_loops is recycling, num_sampling_steps is the
    # requested diffusion step count. ESMFold2 clips the Karras schedule at
    # sigma_max=256, so a requested 100 executes 68 -- requested is not executed here.
    kw = dict(num_loops=args.recycles, num_sampling_steps=args.steps,
              num_diffusion_samples=SAMPLES)
    last_end = None
    for i in range(args.repeat + 1):
        torch.manual_seed(SEED)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if last_end is not None:
            gaps.append(t0 - last_end)
        with torch.no_grad():
            pdb = model.infer_protein_as_pdb(seq, **kw)
        torch.cuda.synchronize()
        last_end = time.perf_counter()
        times.append(last_end - t0)
        p = work / f"fold{i}.pdb"
        p.write_text(pdb if isinstance(pdb, str) else pdb[0])
        preds.append(str(p))

    return dict(summarize(times, args.repeat, gaps), load_s=round(load_s, 2),
                cudnn_sdp_disabled=cudnn_sdp_disabled,
                kernel_counts_total=dict(cueq, **sdpa), predictions=preds,
                fold_kwargs=kw, kernel_backend=backend, msa_rows=0,
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
    ap.add_argument("--esm-module",
                    default="transformers.models.esmfold2.modeling_esmfold2",
                    help="where ESMFold2Model lives; transformers ships it, esm does not")
    ap.add_argument("--esm-repo", default="biohub/ESMFold2")
    ap.add_argument("--esm-backend", default="cuequivariance",
                    choices=["cuequivariance", "fused", "reference"],
                    help="ESMFold2 kernel backend; 'reference' passes None")
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
