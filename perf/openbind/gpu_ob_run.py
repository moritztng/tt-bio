"""One OpenBind-0 rung: N folds of one committed input in a single process, kernel paths COUNTED
and the device half split from the host half.

    /root/venv-ob/bin/python gpu_ob_run.py \
        --spec perf/openbind/inputs/ob_apo_512.spec.json \
        --ckpt /root/ckpt/of3-ob-2025-06-30-174k.pt \
        --work /work/ob_apo_512 --report /root/results/ob_apo_512.json --reps 4

Forked from perf/rf3/gpu_rf3_run.py (the device/host split, the co-tenancy guard) and
scripts/gpu_vs_tt/gpu5_bench.py::run_of3 (which established how to drive the OpenFold3 CLI
without tripping over its defaults). The traps that harness paid for and this one inherits:

  * `--num-diffusion-samples` ships 5. Passed explicitly, never left implicit.
  * `--use-msa-server` ships True and would silently ignore the input and hit ColabFold.
  * `--num-model-seeds` must NOT be passed: any truthy value makes the runner discard the query
    set's seeds and substitute generate_seeds(42, n), so the fold happens at a seed nobody chose.
    The seed the runner actually used is read back out of the output path and asserted.
  * cuEquivariance is opt-in through the runner YAML, not a flag.
  * OF3 catches a per-fold failure, logs it and exits 0. A rung with no CIF is a failure, not a
    fast fold, so the CIF is checked.
  * `experiment_settings.seeds` is what builds the dataset (experiment_runner.py), not the query
    set's `seeds`, so the seed is set in both places.

Why one process, N folds: the shape is fixed, so nothing recompiles, and the 2.3 GB checkpoint is
loaded once. Fold 0 is the cold fold (cuDNN/cueq autotune, allocator growth) and is reported
separately, never folded into the median.

Device time is the number that becomes the port's target. `gpu-reference-device-vs-host-split`:
on the RF3 campaign two independently rented H200s agreed on device time within 1% but differed
up to 2.3x on host featurisation, so publishing wall clock would bake one rented CPU into the
port's bar for months.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import types

COUNTS: dict[str, int] = {}
PHASE: dict[str, list[float]] = collections.defaultdict(list)
FOLD = [-1]                 # index of the fold currently inside predict_step
SHAPES: list[dict] = []     # one entry per fold: token/atom counts read off the real batch


# --------------------------------------------------------------------------------------
# counters -- a flag is not evidence, a call count is
# --------------------------------------------------------------------------------------
# OpenFold3 binds these with `from X import Y`, which copies the reference into its own
# namespace, so the vendor modules MUST be wrapped before openfold3 is imported.
_CUEQ_MODULES = (
    "cuequivariance_ops_torch",
    "cuequivariance_ops_torch.triangle_attention",
    "cuequivariance_torch",
    "cuequivariance_torch.primitives.triangle",
)


def install_cueq_counters() -> None:
    seen = False
    for modname in _CUEQ_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:                                     # noqa: BLE001
            continue
        seen = True
        short = modname.split(".")[-1]
        for name in dir(mod):
            if "triangle" not in name.lower():
                continue
            fn = getattr(mod, name, None)
            if not callable(fn) or isinstance(fn, types.ModuleType):
                continue
            key = "cueq:" + (name if modname == "cuequivariance_ops_torch"
                             else "%s.%s" % (short, name))
            COUNTS[key] = 0

            def make(nm, orig):
                def wrapper(*a, **kw):
                    COUNTS[nm] += 1
                    return orig(*a, **kw)
                return wrapper
            try:
                setattr(mod, name, make(key, fn))
            except Exception:                                 # noqa: BLE001
                COUNTS.pop(key, None)
    if not seen:
        COUNTS["cueq:UNIMPORTABLE"] = -1


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_only(obj, name: str, label: str):
    fn = getattr(obj, name, None)
    if fn is None:
        COUNTS[label] = -1                                    # -1 == not present in this build
        return
    COUNTS[label] = 0

    def wrapper(*a, **kw):
        COUNTS[label] += 1
        return fn(*a, **kw)
    setattr(obj, name, wrapper)


def timed(obj, name: str, phase: str):
    """Count AND time obj.name, cuda-synchronised on both sides.

    Only coarse phases get a sync. The diffusion module is called once per sampling step (200 of
    them per fold) and is counted, not timed: `tt-bio-isolated-op-timing-oversync-inflates-cost`
    is the same mistake in the other direction.
    """
    fn = getattr(obj, name, None)
    if fn is None:
        COUNTS["MISSING:" + phase] = -1
        return
    COUNTS["calls:" + phase] = 0

    def wrapper(*a, **kw):
        COUNTS["calls:" + phase] += 1
        _sync()
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            _sync()
            PHASE["%d/%s" % (FOLD[0], phase)].append(time.perf_counter() - t0)
    setattr(obj, name, wrapper)


# --------------------------------------------------------------------------------------
# the card has to be ours alone
# --------------------------------------------------------------------------------------
def compute_apps() -> list[dict]:
    """Every process holding memory on this GPU, ours included.

    A rented "single GPU" box can share the physical card with another tenant. On the RF3
    campaign that produced a rung reading 9.7-15.2 s where the clean spread is under 4%, at
    578 W against a clean 362 W, and the only thing that showed it was a second compute app
    holding 12486 MiB. Absolute timings from a shared card are void, so the condition is
    recorded per rung rather than assumed away.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        apps = []
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                apps.append({"pid": int(parts[0]), "used_MiB": int(parts[1])})
        return apps
    except Exception as e:                                    # noqa: BLE001
        return [{"error": repr(e)}]


def gpu_static() -> dict:
    q = ("name,driver_version,memory.total,power.limit,clocks.max.sm,compute_cap,"
         "pcie.link.gen.max")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + q,
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        vals = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return dict(zip(q.split(","), vals))
    except Exception as e:                                    # noqa: BLE001
        return {"error": repr(e)}


class PowerSampler(threading.Thread):
    """Power draw at a fixed rung is the cheap co-tenancy detector: nothing but extra work
    explains a big jump in wattage at unchanged input."""

    def __init__(self, period: float = 2.0):
        super().__init__(daemon=True)
        self.period, self.watts, self._stop = period, [], threading.Event()

    def run(self):
        while not self._stop.wait(self.period):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                self.watts.append(float(out.strip().splitlines()[0]))
            except Exception:                                 # noqa: BLE001
                pass

    def stop(self) -> dict:
        self._stop.set()
        self.join(timeout=10)
        if not self.watts:
            return {"n": 0}
        w = sorted(self.watts)
        return {"n": len(w), "median_W": w[len(w) // 2], "max_W": w[-1], "min_W": w[0]}


# --------------------------------------------------------------------------------------
def build_query_set(spec: dict, n: int, seed: int) -> dict:
    """The committed one-query spec, repeated under n names so one process does cold + warm
    folds with no weight reload. Content per query is byte-identical to the committed
    `<name>.of3.json`; only the key differs."""
    chains = []
    for c in spec["chains"]:
        if c["molecule_type"] == "protein":
            chains.append({"molecule_type": "protein", "chain_ids": c["chain_id"],
                           "sequence": c["sequence"]})
        else:
            chains.append({"molecule_type": "ligand", "chain_ids": c["chain_id"],
                           "ccd_codes": c["ccd_codes"]})
    q = {"chains": chains, "use_msas": False, "use_paired_msas": False, "use_main_msas": False}
    return {"seeds": [seed], "queries": {"fold%d" % i: dict(q) for i in range(n)}}


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    mid = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    return {"n": len(s), "median_s": round(mid, 4), "min_s": round(s[0], 4),
            "max_s": round(s[-1], 4), "all_s": [round(x, 4) for x in s]}


def check_output(out_dir: pathlib.Path, seed: int, want_ligand: bool) -> tuple[bool, str]:
    cifs = sorted(out_dir.rglob("*.cif")) + sorted(out_dir.rglob("*.cif.gz"))
    if not cifs:
        return False, ("no CIF under %s: every fold failed inside the runner and the CLI still "
                       "exited 0" % out_dir)
    seeds = sorted({m.group(1) for p in cifs if (m := re.search(r"seed[_-](\d+)", str(p)))})
    if seeds and seeds != [str(seed)]:
        return False, ("folded at seed(s) %r, not %d: the runner overrode the query set, so this "
                       "rung is not comparable" % (seeds, seed))
    import gzip
    p = cifs[0]
    opener = gzip.open if p.suffix == ".gz" else open
    n_atom, bad, hetatm = 0, 0, 0
    with opener(p, "rt") as fh:                               # type: ignore[operator]
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            n_atom += 1
            hetatm += line.startswith("HETATM")
            for tok in line.split()[10:13]:
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if v != v or abs(v) == float("inf"):
                    bad += 1
    if n_atom == 0:
        return False, "%s has 0 atoms" % p.name
    if bad:
        return False, "%s has %d non-finite coordinates" % (p.name, bad)
    if want_ligand and hetatm == 0:
        return False, ("%s has no HETATM: the ligand was dropped, so this rung measured the apo "
                       "protein" % p.name)
    return True, "%d atoms (%d HETATM) in %s, %d CIFs" % (n_atom, hetatm, p.name, len(cifs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--reps", type=int, default=3, help="warm folds; fold 0 is extra and cold")
    ap.add_argument("--samples", type=int, default=1, help="--num-diffusion-samples")
    ap.add_argument("--seed", type=int, default=42, help="OF3's own shipped inference seed")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    spec = json.loads(pathlib.Path(args.spec).read_text())
    want_ligand = any(c["molecule_type"] == "ligand" for c in spec["chains"])
    n_folds = args.reps + 1

    install_cueq_counters()                                   # BEFORE openfold3 is imported
    import torch

    work = pathlib.Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    qpath = work / "queries.json"
    qpath.write_text(json.dumps(build_query_set(spec, n_folds, args.seed), indent=2))

    # cuEquivariance is opt-in through the runner YAML rather than a CLI flag. The seed goes here
    # too: InferenceQuerySet.seeds is parsed but the dataset is built from
    # experiment_settings.seeds, so a query file asking for seed S alone folds at the default.
    ypath = work / "runner.yaml"
    ypath.write_text(
        "experiment_settings:\n  seeds: [%d]\n" % args.seed
        + "model_update:\n  presets: [\"predict\"]\n  custom:\n    settings:\n"
          "      memory:\n        eval:\n          use_cueq_triangle_kernels: true\n")

    # --- phase timers ------------------------------------------------------------------
    runner_mod = importlib.import_module("openfold3.projects.of3_all_atom.runner")
    model_mod = importlib.import_module("openfold3.projects.of3_all_atom.model")
    heads_mod = importlib.import_module("openfold3.core.model.heads.head_modules")
    diff_mod = importlib.import_module("openfold3.core.model.structure.diffusion_module")
    trimul_mod = importlib.import_module(
        "openfold3.core.model.layers.triangular_multiplicative_update")

    Runner = getattr(runner_mod, "OpenFold3AllAtom")
    assert hasattr(Runner, "predict_step"), "no predict_step on OpenFold3AllAtom"

    # predict_step is the device phase: it is `self(batch)` -> trunk + rollout + heads, plus
    # confidence scoring. Featurisation runs in the dataloader and writing runs in the callback,
    # so both land outside it. That boundary is the whole point: device_s is the port's bar and
    # host_s is common cost both arms pay on their own CPU.
    _predict_step = Runner.predict_step

    def predict_step(self, batch, batch_idx):
        FOLD[0] += 1
        shape = {}
        for k, dim in (("token_mask", -1), ("residue_index", -1), ("ref_mask", -1)):
            t = batch.get(k)
            if t is not None and hasattr(t, "shape"):
                shape[k] = list(t.shape)
        for k in ("is_ligand", "is_protein"):
            t = batch.get(k)
            if t is not None and hasattr(t, "sum"):
                shape["n_" + k] = int(t.sum().item())
        SHAPES.append(shape)
        torch.cuda.reset_peak_memory_stats()
        _sync()
        t0 = time.perf_counter()
        try:
            return _predict_step(self, batch, batch_idx)
        finally:
            _sync()
            PHASE["%d/device" % FOLD[0]].append(time.perf_counter() - t0)
            shape["peak_vram_alloc_B"] = int(torch.cuda.max_memory_allocated())
            shape["peak_vram_reserved_B"] = int(torch.cuda.max_memory_reserved())

    Runner.predict_step = predict_step

    timed(getattr(model_mod, "OpenFold3"), "run_trunk", "trunk")
    timed(getattr(model_mod, "OpenFold3"), "_rollout", "rollout")
    timed(getattr(heads_mod, "AuxiliaryHeadsAllAtom"), "forward", "confidence_heads")
    timed(Runner, "_compute_confidence_scores", "confidence_scores")
    count_only(getattr(diff_mod, "DiffusionModule"), "forward", "calls:diffusion_module")
    count_only(trimul_mod, "_cueq_triangle_mult", "of3:_cueq_triangle_mult")

    _sdpa = torch.nn.functional.scaled_dot_product_attention
    COUNTS["torch_sdpa"] = 0

    def sdpa(*a, **kw):
        COUNTS["torch_sdpa"] += 1
        return _sdpa(*a, **kw)
    torch.nn.functional.scaled_dot_product_attention = sdpa

    # --- env ---------------------------------------------------------------------------
    from importlib.metadata import PackageNotFoundError, version
    env = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(),
           "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "python": sys.version.split()[0], "gpu_static": gpu_static(),
           "compute_apps_before": compute_apps()}
    for pkg in ("openfold3", "cuequivariance-torch", "cuequivariance-ops-torch-cu12",
                "cuequivariance", "pytorch-lightning", "numpy", "rdkit", "biotite"):
        try:
            env[pkg] = version(pkg)
        except PackageNotFoundError:
            env[pkg] = None
    try:
        from openfold3.core.kernels.cueq_utils import is_cuequivariance_available
        env["is_cuequivariance_available"] = bool(is_cuequivariance_available())
    except Exception as e:                                    # noqa: BLE001
        env["is_cuequivariance_available"] = repr(e)

    report = {"label": args.label or spec["name"], "spec": args.spec, "spec_content": spec,
              "ckpt": args.ckpt, "reps": args.reps, "n_folds": n_folds,
              "samples": args.samples, "seed": args.seed, "env": env,
              "device_s": None, "wall_s": None, "load_s": None, "host_s": None,
              "phases": {}, "counts": None, "shapes": None, "power": None,
              "compute_apps_after": None, "gpu_exclusive": None, "ok": False, "why": ""}

    def dump():
        report["counts"] = dict(COUNTS)
        report["shapes"] = SHAPES
        per_fold: dict[str, dict] = {}
        for key, xs in PHASE.items():
            fold, phase = key.split("/", 1)
            per_fold.setdefault(fold, {})[phase] = round(sum(xs), 5)
            per_fold[fold][phase + "_n"] = len(xs)
        report["phases"] = per_fold
        p = pathlib.Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2) + "\n")

    # --- fold --------------------------------------------------------------------------
    # --num-model-seeds is deliberately absent, see the module docstring.
    argv = ["predict", "--query-json", str(qpath), "--output-dir", str(work / "out"),
            "--inference-ckpt-path", args.ckpt, "--runner-yaml", str(ypath),
            "--num-diffusion-samples", str(args.samples),
            "--use-msa-server", "false", "--use-templates", "false"]
    print("[ob] argv: %s" % " ".join(argv), flush=True)
    report["cli_argv"] = argv

    power = PowerSampler()
    power.start()
    t0 = time.perf_counter()
    try:
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
        report["wall_s"] = round(time.perf_counter() - t0, 4)
        report["power"] = power.stop()

        dev = [PHASE["%d/device" % i][0] for i in range(n_folds) if PHASE.get("%d/device" % i)]
        report["n_device_calls"] = len(dev)
        report["cold_device_s"] = round(dev[0], 4) if dev else None
        report["device_s"] = summarize(dev[1:])
        for phase in ("trunk", "rollout", "confidence_heads", "confidence_scores"):
            xs = [sum(PHASE.get("%d/%s" % (i, phase), [])) for i in range(1, n_folds)
                  if PHASE.get("%d/%s" % (i, phase))]
            report["warm_" + phase] = summarize(xs)
        warm_dev = report["device_s"].get("median_s")
        # host_s is everything the wall clock spent outside the timed device calls, minus the
        # one-off checkpoint load, divided over the folds. Featurisation dominates it.
        if warm_dev is not None and dev:
            report["host_s"] = round((report["wall_s"] - sum(dev)) / n_folds, 4)
        vram = [s.get("peak_vram_reserved_B", 0) for s in SHAPES[1:]]
        report["peak_vram_reserved_B"] = max(vram) if vram else None
        report["peak_vram_alloc_B"] = max(
            [s.get("peak_vram_alloc_B", 0) for s in SHAPES[1:]] or [0]) or None

        after = compute_apps()
        report["compute_apps_after"] = after
        before = env.get("compute_apps_before") or []
        report["gpu_exclusive"] = (len([a for a in after if "pid" in a]) <= 1
                                   and len([a for a in before if "pid" in a]) <= 1)

        ok, why = check_output(work / "out", args.seed, want_ligand)
        if ok and len(dev) != n_folds:
            ok, why = False, ("predict_step ran %d times, expected %d: some fold was skipped and "
                              "the median is not over what it claims" % (len(dev), n_folds))
        if ok and not report["gpu_exclusive"]:
            ok, why = False, ("GPU NOT EXCLUSIVE: compute apps before=%r after=%r, timings on a "
                              "shared card are void" % (before, after))
        report["ok"], report["why"] = ok, why
    except Exception:                                         # noqa: BLE001
        import traceback
        report["power"] = power.stop()
        report["ok"] = False
        report["why"] = traceback.format_exc()[-6000:]
        dump()
        raise
    finally:
        dump()
    print("[ob] %s ok=%s device_median=%s host=%s why=%s"
          % (report["label"], report["ok"], report["device_s"].get("median_s"),
             report["host_s"], report["why"]), flush=True)


if __name__ == "__main__":
    main()
