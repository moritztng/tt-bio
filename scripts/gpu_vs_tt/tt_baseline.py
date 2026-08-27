#!/usr/bin/env python3
"""TT baseline leg of the GPU-vs-Tenstorrent Protenix-family head-to-head.

Measures per-fold latency on one Blackhole card, in-process, at production
config: a target YAML (MSA on), 200 diffusion sampling steps / 1 sample / seed 0,
and the checkpoint's own recycle count (protenix-v2 and OpenDDE 10, protenix-v1 4;
--recycling_steps forces a matched arm). The model is loaded once; one cold fold
absorbs first-kernel compile (reported separately, never in the warm numbers);
then N warm folds give min/median/max. The committed alignment is seeded into
the MSA cache as {seq_hash}.a3m before any fold, so MSA search never runs at all
-- not in the timed region and not in the cold fold.

Two targets share this harness, which is what makes the scaling comparison
possible: examples/prot.yaml (117 aa) and examples/prot300.yaml (CDK2, 298 aa).
Both use a 35-sequence alignment, so token count is the only variable.

Usage (on a TT host, card pinned via TT_VISIBLE_DEVICES):

    TT_VISIBLE_DEVICES=1 python3 scripts/gpu_vs_tt/tt_baseline.py \
        --model protenix-v2 --repeat 3 --out tt_protenix.json

    TT_VISIBLE_DEVICES=1 python3 scripts/gpu_vs_tt/tt_baseline.py \
        --model protenix-v2 --repeat 3 --target examples/prot300.yaml \
        --msa-a3m scripts/gpu_vs_tt/fixtures/prot300.a3m --label "298 aa" \
        --out tt_protenix_300.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Appended, never prepended: a driver may run this script out of one tree while
# PYTHONPATH points tt_bio at another, and prepending would silently win that fight.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from tt_bio.main import PROTENIX_FAMILY, _resolve_recycling_steps  # noqa: E402

# Production config, identical to what the GPU leg must run (fairness contract).
# The recycle count is NOT one of these: it is a per-checkpoint property, so it comes
# from tt_bio.main.RECYCLING_STEPS, the single table the CLI reads. A module constant
# here would fold protenix-v1 at protenix-v2's 10 and label the result a default-settings
# number -- v1's own config ships 4. --recycling_steps overrides it for a matched-settings
# arm, and the resolved value is written into the result JSON, because a perf record that
# does not state its own recycle count cannot be compared to anything.
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1
SEED = 0

# Models this harness can fold. Narrower than tt_bio.main.PREDICT_MODELS on purpose:
# build_fold seeds a single protein alignment into the MSA cache and drives the shared
# Protenix-class predict_one, so it covers the Protenix family and the OpenDDE
# checkpoints that ride the same path. Derived rather than hand-typed -- a hardcoded
# list here is what kept protenix-v1 unmeasurable after it was registered everywhere else.
BASELINE_MODELS = tuple(PROTENIX_FAMILY) + ("opendde", "opendde-abag")


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _ttnn_version() -> str:
    try:
        from importlib.metadata import version
        return version("ttnn")
    except Exception:
        return "unknown"


def _card_info() -> dict:
    """Board type, PCI subsystem and AICLK for the card this run will use. No device open.

    Board type comes from ``perf_regression.detect_card_type()``, the repo's canonical
    detector and the key the perf gate's per-card baselines are stored under, so a perf
    artifact and a gate baseline name the same part the same way. It never returns None:
    tt-smi first, then the kernel sysfs subsystem as ``unknown:<sub>``.

    That fallback is the point. ``tt-smi -s`` hangs on a box that is already folding
    (measured on tt-quietbox 2026-08-23: rc=124 at a 60 s timeout with four jobs on the
    cards), which is exactly when a benchmark runs, so the previous tt-smi-only version
    silently left ``card_type: null`` in every artifact it wrote. All six published
    perf-page prediction cells were measured on tt-quietbox2 and carried on tt-bio.com as
    Blackhole p150a for weeks, because nothing in their provenance recorded the board.
    ``sysfs_subsystem`` is stored raw next to it: it always reads, and it is what settles
    the question afterwards.
    """
    info: dict = {}
    visible = (os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "tt_bio_perf_regression", REPO_ROOT / "scripts" / "perf_regression.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        info["card_type"] = mod.detect_card_type()
        info["sysfs_subsystem"] = mod._sysfs_subsystem_device(visible)
    except Exception as e:
        info["card_type_error"] = repr(e)
    tt_smi = None
    for c in (Path(sys.executable).parent / "tt-smi",
              Path.home() / ".local" / "bin" / "tt-smi",
              Path("/usr/local/bin/tt-smi")):
        if c.is_file() and os.access(c, os.X_OK):
            tt_smi = str(c)
            break
    if tt_smi is None:
        return info
    try:
        out = subprocess.run([tt_smi, "-s"], capture_output=True, text=True, timeout=20)
        devs = json.loads(out.stdout).get("device_info", [])
        idx = min(int(visible), len(devs) - 1) if devs and visible.isdigit() else 0
        tel = devs[idx].get("telemetry", {}) or {}
        if tel.get("aiclk"):
            info["aiclk_mhz"] = tel["aiclk"]
    except Exception:
        pass
    return info


def _grid() -> list | None:
    """Tensix compute grid of the open device, e.g. [13, 10] on a p150a and [11, 10] on the
    p300c chips in a TT-QuietBox 2. Two boards of the same Blackhole generation can differ
    here by 18 % of the cores, and a fold time without it cannot be compared across boxes."""
    try:
        from tt_bio.tenstorrent import get_device
        g = get_device().compute_with_storage_grid_size()
        return [g.x, g.y]
    except Exception:
        return None


def seed_msa_cache(target: Path, a3m: Path, msa_dir: Path) -> int:
    """Install ``a3m`` as the cached alignment for ``target``'s chain, so the timed
    folds hit the cache and no MSA search ever runs. Same contract the 117-aa run
    used ({sha256(seq)[:16]}.a3m in msa_dir); asserts the a3m's query row is the
    target's own sequence, which is what makes "identical bytes both sides" a
    checked fact rather than a claim.
    """
    import hashlib

    from tt_bio.main import _read_bio_chains

    chains = _read_bio_chains(target)
    assert len(chains) == 1, f"{target} is not a monomer: {len(chains)} chains"
    seq = chains[0][1]
    text = a3m.read_text()
    rows = text.split("\n")
    assert rows[1] == seq, f"{a3m} query row does not match {target}'s sequence"
    msa_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(seq.encode()).hexdigest()[:16]
    (msa_dir / f"{h}.a3m").write_text(text)
    return text.count(">")


# Named-function phase buckets for the host/device split. Which callee of
# ``_WorkerState._predict_<model>_one`` is host work, which is accelerator work, which
# is the host->device copy. Bucket by NAMED FUNCTION, never by region: OpenFold3 runs
# ``run_input_atom_encoder`` on the card from inside the pre-``fold`` region, so an
# "everything before the fold is host" bracket books device work as host.
#
#   ("mod", "<module>:<attr>", bucket)  -- patch the attribute on the source module.
#       Every callee below is imported function-locally inside the ``_predict_*_one``
#       that calls it, so the name is resolved at call time and the patch is seen.
#   ("state", "<attr path>", bucket)    -- patch an attribute on the _WorkerState (or
#       on something it holds). Needed where the callee is bound at load time:
#       ``state.prepare`` is a ``partial`` built in ``bind_run`` (worker.py:566), so
#       patching ``tt_bio.main.prepare_features`` after the bind measures zero.
#
# Whatever the brackets miss lands in ``residual``. For all five models the only
# device entry points are ``model.fold`` / ``model.predict_step`` and OpenFold3's
# ``run_input_atom_encoder``, all named here, so the residual is host glue.
PHASES: dict[str, list[tuple[str, str, str]]] = {
    "boltz2": [
        ("state", "prepare", "host"),                            # worker.py:640
        ("mod", "tt_bio.main:to_batch", "transfer"),             # worker.py:641
        ("state", "model.predict_step", "device"),               # worker.py:644
        ("mod", "tt_bio.main:write_result", "host"),             # worker.py:645
    ],
    "esmfold2": [
        ("mod", "tt_bio.main:_read_protein_chains", "host"),           # worker.py:665
        ("mod", "tt_bio.esmfold2_runtime:resolve_msa", "host"),        # worker.py:705
        ("mod", "tt_bio.esmfold2_runtime:fold_complex", "device"),     # worker.py:707
        ("mod", "tt_bio.main:_write_structure", "host"),               # worker.py:727
    ],
    "protenix-v2": [
        ("mod", "tt_bio.protenix_data:build_complex_features", "host"),  # worker.py:931
        ("state", "model.fold", "device"),                               # worker.py:1009
        ("mod", "tt_bio.main:_write_protenix_structure", "host"),        # worker.py:963
    ],
    "opendde": [
        ("mod", "tt_bio.protenix_data:build_complex_features", "host"),  # worker.py:835
        ("state", "model.fold", "device"),                               # worker.py:847
        ("mod", "tt_bio.main:_write_protenix_structure", "host"),        # worker.py:870
    ],
    "openfold3": [
        ("mod", "tt_bio.main:_read_bio_chains", "host"),                       # :1239
        ("mod", "tt_bio.openfold3_data:resolve_openfold3_msas", "host"),       # :1305
        ("mod", "tt_bio.openfold3_data:build_openfold3_features", "host"),     # :1343
        ("mod", "tt_bio.openfold3_data:make_openfold3_msa_features", "host"),  # :1350
        ("mod", "tt_bio.openfold3_host_prep:derive_block_aux", "host"),        # :1352
        ("mod", "tt_bio.openfold3_host_prep:derive_template_feat", "host"),    # :1353
        ("mod", "tt_bio.openfold3_host_prep:dedup_template_slots", "host"),    # :1353
        ("mod", "tt_bio.openfold3_host_prep:derive_relpos", "host"),           # :1355
        ("mod", "tt_bio.openfold3_host_prep:ref_atom_embed", "host"),          # :1363
        ("mod", "tt_bio.openfold3_host_prep:run_input_atom_encoder", "device"),  # :1359, ON CARD
        ("state", "model.fold", "device"),                                     # :1389
        ("mod", "tt_bio.worker:_write_atom_array_structure", "host"),          # :1409
    ],
}


class Instrument:
    """Wall-clock brackets around the named phases of one model's ``_predict_*_one``.

    ``on()`` / ``off()`` swap the wrappers in and out, so a single process can
    alternate an instrumented arm against the uninstrumented published one and prove
    the instrument does not move the wall it measures. Whole-function brackets only:
    nothing here adds a device sync, because a per-op sync roughly doubles the cost it
    is measuring. ``model.fold`` returns host tensors, so its bracket is already synced
    at the boundary.

    Patches this process only; production code is untouched.
    """

    def __init__(self, model: str, state):
        import importlib

        if model not in PHASES:
            raise ValueError(f"no phase table for {model!r}; have {sorted(PHASES)}")
        self.model = model
        self._sites = []
        for kind, target, bucket in PHASES[model]:
            if kind == "mod":
                modname, attr = target.split(":")
                owner = importlib.import_module(modname)
            else:
                owner, *path = state, *target.split(".")
                for step in path[:-1]:
                    owner = getattr(owner, step)
                attr = path[-1]
            self._sites.append((owner, attr, getattr(owner, attr), bucket, target))
        self._cur: dict[str, float] = {}
        self._per_fn: dict[str, dict] = {}
        self._depth = 0
        self.active = False

    def _wrap(self, fn, bucket: str, name: str):
        def timed(*a, **k):
            outer = self._depth == 0
            self._depth += 1
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                dt = time.perf_counter() - t
                self._depth -= 1
                # One bracket can sit inside another: OpenFold3's run_input_atom_encoder
                # calls ref_atom_embed (openfold3_host_prep.py:227), and both are in the
                # table. Charge the buckets from the OUTERMOST bracket only, so the
                # nested call lands in the caller's bucket and is not counted twice.
                # Untracked, that showed up as a negative residual of 0.029 s.
                if outer:
                    self._cur[bucket] = self._cur.get(bucket, 0.0) + dt
                row = self._per_fn.setdefault(name, {"s": 0.0, "n": 0, "nested": 0})
                row["s"] += dt
                row["n"] += 1
                row["nested"] += not outer
        return timed

    def on(self) -> None:
        for owner, attr, orig, bucket, name in self._sites:
            setattr(owner, attr, self._wrap(orig, bucket, name))
        self.active = True

    def off(self) -> None:
        for owner, attr, orig, _bucket, _name in self._sites:
            setattr(owner, attr, orig)
        self.active = False

    def row(self, total: float) -> dict:
        """The phase record for the fold that just ran; resets the accumulators.

        ``residual`` is what the brackets missed, and it is host glue by construction
        (see PHASES), so it cannot go negative: nested brackets charge the outermost
        bucket once. ``per_fn`` keeps the inclusive per-call time, its call count and
        how many of those calls were nested inside another bracket -- ``n`` separates
        "this patch never fired" from "it fired and cost nothing", and ``nested`` says
        how much of that time is already inside another row.
        """
        r = {b: round(self._cur.get(b, 0.0), 4) for b in ("host", "device", "transfer")}
        r["total"] = round(total, 4)
        r["residual"] = round(total - sum(self._cur.values()), 4)
        r["per_fn"] = {k: {"s": round(v["s"], 4), "n": v["n"], "nested": v["nested"]}
                       for k, v in self._per_fn.items()}
        self._cur.clear()
        self._per_fn.clear()
        return r


def build_fold(model: str, msa_dir: Path, target: Path, a3m: Path,
               samples: int = DIFFUSION_SAMPLES, hoist: bool = False,
               instrument: bool = False, fast: bool = False, trace: bool = False,
               recycling_steps: int | None = None):
    """Open the card, load the model, seed the MSA cache; return ``(one_fold, meta)``.

    Split out of ``measure`` so the multi-card fan-out driver (``tt_concurrency.py``)
    folds through exactly the same path as the single-card latency baseline. Aggregate
    throughput and per-fold latency have to come from the same fold or the box-level
    number cannot be checked against the per-card one.

    ``hoist`` changes the timed region to match the GPU harness exactly: featurization
    (chain read, MSA resolve, build_complex_features) runs ONCE up front and the CIF
    write is suppressed, so a timed fold is ``model.fold`` only -- the same boundary as
    the GPU leg's ``runner.predict`` + ``cuda.synchronize()``. The first call to
    ``one_fold`` is still the full ``predict_one`` (cold fold: validates the MSA cache
    and warms every kernel); only the timed calls switch to the hoisted region. The raw
    ``predict_one`` boundary stays the default; the committed numbers used it.

    ``instrument`` (raw mode only) brackets the named phases of the model's
    ``_predict_*_one`` -- see ``PHASES`` above -- and appends one
    ``{host, device, transfer, total, residual, per_fn}`` entry per fold to
    ``meta["phase_times"]``, so the host/device split is measured on the same folds the
    raw number comes from. Monkeypatches attributes inside this process only;
    production code is untouched. Supported for every model in ``PHASES``. A driver
    that needs to alternate arms in one process builds an ``Instrument`` itself and
    toggles it around folds instead of passing this flag.
    """
    import torch  # noqa: F401
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device, arch_name
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E

    if hoist and model != "protenix-v2":
        raise ValueError("hoist is protenix-v2 only (the disputed model)")

    _noop = lambda *a, **k: None
    _E.set_progress(_noop)

    # --trace replays a captured ttnn trace of the per-step diffusion device graph, and
    # opendde.py:481 refuses to fold with trace=True unless the device carries a region.
    get_device(trace_region_size=(1 << 30) if trace else 0)  # open the chip once (lease enforced here)
    hw = arch_name()

    work = Path(tempfile.mkdtemp(prefix=f"ttbase-{model}-"))
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    recycles = _resolve_recycling_steps(recycling_steps, model)

    cfg = dict(
        model=model, fast=fast, output_format="cif",
        recycling_steps=recycles, sampling_steps=SAMPLING_STEPS,
        diffusion_samples=samples, seed=SEED, trace=trace,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=True, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None,
        msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
    )
    _ensure_local_artifacts(cfg)
    n_msa = seed_msa_cache(target, a3m, msa_dir)

    state = _WorkerState("tenstorrent")
    t_load = time.perf_counter()
    state.load_model(cfg)
    load_s = time.perf_counter() - t_load
    state.bind_run("ttbase", cfg)
    state.pfn = _noop

    job_cfg = dict(cfg)

    phase_times: list[dict] = []
    inst = Instrument(model, state) if instrument else None
    if inst is not None:
        inst.on()

    hoisted = None
    if hoist:
        from tt_bio.main import (_read_bio_chains, _read_bio_constraints,
                                 _resolve_a3m_text)
        from tt_bio.protenix_data import build_complex_features

        chains = _read_bio_chains(target)
        bonds = _read_bio_constraints(target)
        chain_specs = [(cseq, _resolve_a3m_text(spec, cseq, msa_dir)
                        if mt == "protein" else None, mt)
                       for _cid, cseq, spec, mt in chains]
        t_feat = time.perf_counter()
        feats_h = build_complex_features(
            chain_specs, mol_dir=cfg.get("mol_dir"),
            chain_ids=[cid for cid, _s, _sp, _mt in chains], bonds=bonds)
        feat_once_s = time.perf_counter() - t_feat
        n_res = sum(len(cseq) for _c, cseq, _s, mt in chains if mt != "ligand")

        def hoisted():
            t0 = time.perf_counter()
            with torch.no_grad(), state._maybe_ref_bf16():
                # Fresh top-level mapping per fold. The GPU leg had to do this because
                # protenix's forward deletes the MSA keys out of the feature dict in
                # place; if this fold does the same, reusing one dict would quietly
                # fold folds 2..n without an MSA and report a fast, wrong number. The
                # pLDDT check in tt_concurrency is what actually proves it did not.
                _coords, conf = state.model.fold(
                    dict(feats_h), n_step=SAMPLING_STEPS, n_sample=samples, seed=SEED,
                    progress_fn=_noop, return_confidence=True,
                    n_cycles=recycles,
                    max_parallel_samples=cfg.get("max_parallel_samples"), trace=False)
            confs = conf if isinstance(conf, list) else [conf]
            metrics = {"plddt": confs[0]["plddt"], "msa": True,
                       "n_tokens": int(feats_h["restype"].shape[0]),
                       "n_residues": n_res, "samples": samples}
            return time.perf_counter() - t0, metrics

    cold_done = {"v": False}

    def one_fold():
        job_cfg["struct_dir"] = str(struct_dir)
        for p in struct_dir.glob("*"):
            p.unlink()
        if hoist and cold_done["v"]:
            return hoisted()
        t0 = time.perf_counter()
        metrics, _best, _feats = state.predict_one(target, job_cfg)
        total = time.perf_counter() - t0
        cold_done["v"] = True
        if inst is not None and inst.active:
            phase_times.append(inst.row(total))
        return total, metrics

    return one_fold, dict(hardware=hw, grid=_grid(), recycling_steps=recycles,
                          load_s=round(load_s, 2), n_msa=n_msa,
                          msa_dir=str(msa_dir), diffusion_samples=samples,
                          job_cfg=job_cfg, struct_dir=str(struct_dir),
                          timed_region=("model.fold only (featurization hoisted, "
                                        "CIF write suppressed)" if hoist else
                                        "predict_one (featurize + fold + CIF write)"),
                          phase_times=phase_times if instrument else None,
                          **_card_info()), state


def measure(model: str, repeat: int, msa_dir: Path, out_path: Path,
            target: Path, a3m: Path, label: str, fast: bool = False,
            keep_cif: Path | None = None, trace: bool = False,
            ab_env: str | None = None, ab_values: tuple[str, str] = ("1", "0"),
            recycling_steps: int | None = None) -> dict:
    one_fold, meta, state = build_fold(model, msa_dir, target, a3m, fast=fast, trace=trace,
                                       recycling_steps=recycling_steps)
    from tt_bio.tenstorrent import cleanup
    hw, load_s, n_msa = meta["hardware"], meta["load_s"], meta["n_msa"]

    # Cold fold: first-kernel compile. Never counted in the warm numbers. The MSA
    # cache was seeded above, so no fold here -- cold or warm -- runs a search.
    # With --ab-env each arm gets its own cold fold: the arms run different shapes, so one
    # arm's cold fold does not compile the other's kernels or fill its program cache.
    if ab_env:
        os.environ[ab_env] = ab_values[0]
    cold_s, cold_metrics = one_fold()
    assert cold_metrics.get("msa"), "fold ran without an MSA -- cache seeding failed"
    msa_hits = sorted(p.name for p in msa_dir.glob("*.a3m"))

    def _fold_record(t, m, arm=None):
        """Per-fold provenance: the CIF hash the arm produced, its plDDT and the host load.

        An A/B whose arms differ in numerics must show what each arm actually built, and the
        load at the fold is what made an earlier 512 aa figure unusable. struct_dir is cleared
        at the START of the next fold, so this reads it while it is still the current one.
        """
        import hashlib
        cifs = {}
        for f in sorted(Path(meta["struct_dir"]).glob("*.cif")):
            cifs[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        if keep_cif is not None:
            # struct_dir is cleared at the start of the next fold, so the copy happens here or
            # never. One directory per arm, named by the caller, so an RMSD control has a file.
            import shutil
            d = keep_cif if arm is None else keep_cif / f"arm_{arm}"
            d.mkdir(parents=True, exist_ok=True)
            for f in sorted(Path(meta["struct_dir"]).glob("*.cif")):
                shutil.copy2(f, d / f.name)
        rec = {"s": round(t, 3), "plddt": m.get("plddt"), "cif_sha256": cifs,
               "loadavg": open("/proc/loadavg").read().split()[:3]}
        if arm is not None:
            rec["arm"] = arm
        return rec

    fold_recs = [dict(_fold_record(cold_s, cold_metrics, ab_values[0] if ab_env else None),
                      arm_fold="cold")]
    times = []
    ab = None
    if ab_env:
        # PAIRED, INTERLEAVED arms in ONE process, which is the only way to resolve a delta
        # smaller than the spread between two separate invocations of this script: the
        # invocation-to-invocation offset (0.29 % on OpenDDE at 512 aa) is shared by both halves
        # of a pair and cancels in the difference, while the fold-to-fold spread inside an arm is
        # ~0.06 %. The within-pair order alternates so a monotone drift across the run cannot
        # masquerade as an effect.
        os.environ[ab_env] = ab_values[1]
        cold_b_s, cold_b_m = one_fold()
        fold_recs.append(dict(_fold_record(cold_b_s, cold_b_m, ab_values[1]), arm_fold="cold"))
        arm_times = {v: [] for v in ab_values}
        for i in range(repeat):
            order = ab_values if i % 2 == 0 else tuple(reversed(ab_values))
            for v in order:
                os.environ[ab_env] = v
                ts, m = one_fold()
                arm_times[v].append(ts)
                fold_recs.append(dict(_fold_record(ts, m, v), arm_fold=f"warm{i}"))
        deltas = [round(a - b, 3) for a, b in zip(arm_times[ab_values[0]],
                                                  arm_times[ab_values[1]])]
        ab = {"env": ab_env, "values": list(ab_values),
              "arms": {v: {"warm_times_s": [round(x, 3) for x in arm_times[v]],
                           "warm_median_s": round(sorted(arm_times[v])[len(arm_times[v]) // 2], 3)}
                       for v in ab_values},
              "paired_delta_s": deltas,
              "paired_delta_median_s": round(sorted(deltas)[len(deltas) // 2], 3),
              "paired_delta_mean_s": round(sum(deltas) / len(deltas), 4),
              "note": f"delta = {ab_env}={ab_values[0]} minus {ab_env}={ab_values[1]}, "
                      "same process, alternating within-pair order, per-arm cold fold discarded"}
        times = arm_times[ab_values[0]]
        print(f"[{model}] paired {ab_env} {ab_values[0]}-vs-{ab_values[1]} deltas {deltas} s, "
              f"median {ab['paired_delta_median_s']:+.3f} s", file=sys.stderr)
    else:
        for _ in range(repeat):
            t, _m = one_fold()
            times.append(t)
            fold_recs.append(_fold_record(t, _m))

    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    result = dict(
        model=model, side="tenstorrent",
        hardware=hw, machine=socket.gethostname(),
        visible_devices=os.environ.get("TT_VISIBLE_DEVICES", "0"),
        **_card_info(),
        tt_bio_git=_git_sha(), ttnn_version=_ttnn_version(),
        input=f"{target.name} ({label}, single protein, MSA on)",
        target=str(target), n_msa=n_msa,
        n_tokens=cold_metrics.get("n_tokens"), n_residues=cold_metrics.get("n_residues"),
        msa_cache=msa_hits,
        recycling_steps=meta["recycling_steps"], sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES, seed=SEED,
        load_s=round(load_s, 2), cold_s=round(cold_s, 2),
        warm_times_s=[round(t, 3) for t in times],
        warm_min_s=round(times_sorted[0], 3), warm_median_s=round(median, 3),
        warm_max_s=round(times_sorted[-1], 3),
        cold_metrics=cold_metrics,
        warm_folds=fold_recs, ab=ab,
        runtime_root=os.environ.get("TT_METAL_RUNTIME_ROOT", "<unset>"),
        kernel_cache=os.environ.get("TT_METAL_CACHE", "<unset>"),
        uptime=subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
        date=time.strftime("%Y-%m-%d"),
    )
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[{model}] warm median {median:.2f}s "
          f"(min {times_sorted[0]:.2f} / max {times_sorted[-1]:.2f}), "
          f"cold {cold_s:.1f}s, load {load_s:.0f}s", file=sys.stderr)
    state.reset()
    cleanup()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(BASELINE_MODELS))
    ap.add_argument("--recycling_steps", type=int, default=None,
                    help="override the per-model default from tt_bio.main.RECYCLING_STEPS "
                         "(protenix-v1 4, protenix-v2/opendde 10); use it to express a "
                         "matched-settings arm across checkpoints")
    ap.add_argument("--repeat", type=int, default=3, help="timed warm folds (default 3)")
    ap.add_argument("--msa-dir", type=Path,
                    default=Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser())
    ap.add_argument("--target", type=Path, default=REPO_ROOT / "examples" / "prot.yaml",
                    help="single-chain target YAML (default: the 117-aa prot.yaml)")
    ap.add_argument("--msa-a3m", type=Path, default=FIXTURES / "prot117.a3m",
                    help="committed alignment, seeded into the MSA cache before folding")
    ap.add_argument("--label", default="117 aa",
                    help="size label recorded in the result JSON")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fast", action="store_true",
                    help="fold with the shipped --fast block-fp8 path (not the default arm)")
    ap.add_argument("--trace", action="store_true",
                    help="replay a captured ttnn trace of the diffusion step (reserves 1 GiB)")
    ap.add_argument("--keep-cif", type=Path, default=None,
                    help="copy each fold's CIF here, so an arm can be scored against another")
    ap.add_argument("--ab-env", default=None, metavar="NAME",
                    help="run two arms of NAME in ONE process, interleaved and paired, and "
                         "report the per-pair difference. --repeat becomes the number of PAIRS")
    ap.add_argument("--ab-values", default="1,0", metavar="ON,OFF",
                    help="the two values --ab-env alternates (default 1,0)")
    args = ap.parse_args()
    av = tuple(v.strip() for v in args.ab_values.split(","))
    assert len(av) == 2, "--ab-values takes exactly two comma-separated values"
    measure(args.model, args.repeat, args.msa_dir, args.out,
            args.target, args.msa_a3m, args.label, fast=args.fast, keep_cif=args.keep_cif,
            trace=args.trace, ab_env=args.ab_env, ab_values=av,
            recycling_steps=args.recycling_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
