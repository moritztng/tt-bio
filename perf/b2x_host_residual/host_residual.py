#!/usr/bin/env python3
"""The 3.212 s of the Boltz-2 512 aa fold that issues 79 ttnn calls and moves no device bytes.

`b2x-op-cost-curve` bracketed the fold by `tt_bio.tenstorrent` module class and found that
12.882 + 7.151 + 0.465 = 20.498 s of a 23.710 s fold is inside one of those classes. The other
**3.212 s is host code with zero device work**, and nothing in the 2x campaign ever looked at it,
because it was never a row in anyone's table -- it was the gap left over after the rows.

This file makes it rows. Three instruments, each named in the output so no row's provenance is
ambiguous:

  brackets  wall-clock inclusive/exclusive seconds per named region, from `time.perf_counter`.
            Wall, not `thread_time`: on this stack the calling thread SPINS while it waits for
            dispatch-queue room, so CPU time reads as host work when it is device wait, and that
            artifact is exactly what made a published verdict wrong on 2026-09-11. The residual
            has no device work in it, so wall clock is both honest and sufficient.
            The region tree covers `predict_one` completely, so the leftover is measured, not
            inferred: `unattributed = predict_one - (every top-level region)`.

  sampler   a 1 ms stack sampler on the main thread (`sys._current_frames`), giving self-time per
            source line WITHIN a region. Perturbing (~2-4 %, it takes the GIL), so it runs in its
            OWN fold and only ever supplies *shares*; every second in the table comes from the
            bracket fold. Reported shares, never seconds, unless the two folds agree.

  gc        `gc.callbacks` timing. A hypothesis the other two instruments cannot see: the sampler
            loop allocates ~10 float32 tensors per step against a resident graph of a full
            Boltz-2 plus every cached ttnn tensor, and a gen-2 pass traverses all of it. GC time
            shows up inside whatever frame happened to allocate, so a stack profile smears it
            across the whole loop instead of naming it.

Phases:
  plain    N plain folds: wall, plDDT, CIF sha256. The A/A floor and the parity reference.
  attrib   one fold with the region tree installed. The seconds table.
  sample   one fold with the stack sampler + gc timing installed. The within-region shares.
  ab       interleaved A/B of a lever, ABABAB, one process, one device open.

No model code is changed by `plain`/`attrib`/`sample`; every patch is a timing wrapper that
forwards its arguments and returns its callee's value. `ab` toggles a lever through the env.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics as st
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {}
OUT_PATH: Path | None = None
CUR = "(root)"          # current bracket path, read by the sampler and the gc callback


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def loadavg():
    return open("/proc/loadavg").read().split()[:3]


def cif_sha(struct_dir: Path) -> dict:
    out = {}
    for p in sorted(Path(struct_dir).rglob("*")):
        if p.is_file() and p.suffix in (".cif", ".pdb"):
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# =============================================================================================
# instrument 1: wall-clock region brackets
# =============================================================================================
class Regions:
    """Inclusive/exclusive wall per named region, nested by call path.

    No `synchronize_device` anywhere. Inside the sampler loop the data dependency already
    serialises host and device (the denoiser hands a torch tensor back, so the device has
    finished before the next host line runs), and a sync at a region boundary would change the
    thing being measured. The top-level regions are checked against a plain fold's wall instead.
    """

    def __init__(self):
        self.stack: list[str] = []
        self.incl: dict[str, float] = defaultdict(float)
        self.n: dict[str, int] = defaultdict(int)
        self.child: dict[str, float] = defaultdict(float)
        self._undo: list = []
        self.notes: list[str] = []

    def patch(self, owner, attr, label):
        try:
            orig = getattr(owner, attr)
        except AttributeError:
            self.notes.append(f"missing: {owner!r}.{attr} -- region '{label}' not measured")
            return False
        if not callable(orig):
            self.notes.append(f"not callable: {owner!r}.{attr}")
            return False

        def w(*a, **k):
            return self.call(label, orig, a, k)

        try:
            setattr(owner, attr, w)
        except Exception as e:                                             # noqa: BLE001
            self.notes.append(f"cannot patch {owner!r}.{attr}: {e}")
            return False
        # Remember whether the name lived in the instance/module dict or came from a class, so
        # `remove` puts it back the way it was instead of leaving a bound method behind.
        own = attr in getattr(owner, "__dict__", {})
        self._undo.append((owner, attr, orig, own))
        return True

    def call(self, label, fn, a, k):
        global CUR
        path = "/".join(self.stack + [label])
        self.stack.append(label)
        CUR = path
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            self.stack.pop()
            CUR = "/".join(self.stack) or "(root)"
            self.incl[path] += dt
            self.n[path] += 1
            if self.stack:
                self.child["/".join(self.stack)] += dt

    def remove(self):
        for owner, attr, orig, own in reversed(self._undo):
            if own:
                setattr(owner, attr, orig)
            else:
                try:
                    delattr(owner, attr)
                except AttributeError:
                    setattr(owner, attr, orig)
        self._undo = []

    def table(self):
        rows = {}
        for path in sorted(self.incl):
            incl = self.incl[path]
            rows[path] = {
                "calls": self.n[path],
                "incl_s": round(incl, 5),
                "excl_s": round(incl - self.child.get(path, 0.0), 5),
                "ms_per_call": round(1e3 * incl / max(self.n[path], 1), 5),
            }
        return rows


# =============================================================================================
# instrument 2: 1 ms main-thread stack sampler
# =============================================================================================
class StackSampler(threading.Thread):
    def __init__(self, period=0.001, repo=str(ROOT)):
        super().__init__(daemon=True)
        self.period = period
        self.repo = repo
        self.main = threading.main_thread().ident
        self.stop = threading.Event()
        self.leaf: Counter = Counter()       # innermost python frame, any file
        self.site: Counter = Counter()       # innermost frame inside the repo
        self.region: Counter = Counter()     # bracket path only
        self.total = 0
        self.overrun = 0

    def _keys(self, fr):
        leaf = f"{Path(fr.f_code.co_filename).name}:{fr.f_code.co_name}:{fr.f_lineno}"
        site = None
        f = fr
        depth = 0
        while f is not None and depth < 400:
            fn = f.f_code.co_filename
            if fn.startswith(self.repo) and "/_vendor/" not in fn:
                site = f"{Path(fn).name}:{f.f_code.co_name}:{f.f_lineno}"
                break
            f = f.f_back
            depth += 1
        return leaf, site or "(outside repo)"

    def run(self):
        nxt = time.perf_counter()
        while not self.stop.is_set():
            fr = sys._current_frames().get(self.main)
            if fr is not None:
                r = CUR
                leaf, site = self._keys(fr)
                self.leaf[(r, leaf)] += 1
                self.site[(r, site)] += 1
                self.region[r] += 1
                self.total += 1
            nxt += self.period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                self.overrun += 1
                nxt = time.perf_counter()

    def report(self, top=60):
        def fold(c):
            return [{"region": r, "at": k, "n": n, "share_of_region": None}
                    for (r, k), n in c.most_common(top)]
        reg = dict(self.region.most_common())
        site = fold(self.site)
        for row in site:
            row["share_of_region"] = round(row["n"] / max(reg.get(row["region"], 1), 1), 4)
        leaf = fold(self.leaf)
        for row in leaf:
            row["share_of_region"] = round(row["n"] / max(reg.get(row["region"], 1), 1), 4)
        return {"period_s": self.period, "samples": self.total, "overruns": self.overrun,
                "by_region": reg, "by_repo_site": site, "by_leaf_frame": leaf}


# =============================================================================================
# instrument 3: gc.callbacks timing
# =============================================================================================
class GCTimer:
    def __init__(self):
        self.t0 = None
        self.where = "(root)"
        self.by_gen: dict[int, float] = defaultdict(float)
        self.n_gen: dict[int, int] = defaultdict(int)
        self.by_region: dict[str, float] = defaultdict(float)
        self.collected = 0

    def cb(self, phase, info):
        if phase == "start":
            self.t0 = time.perf_counter()
            self.where = CUR
        elif self.t0 is not None:
            dt = time.perf_counter() - self.t0
            g = info.get("generation", -1)
            self.by_gen[g] += dt
            self.n_gen[g] += 1
            self.by_region[self.where] += dt
            self.collected += info.get("collected", 0)
            self.t0 = None

    def on(self):
        gc.callbacks.append(self.cb)

    def off(self):
        try:
            gc.callbacks.remove(self.cb)
        except ValueError:
            pass

    def report(self):
        return {"total_s": round(sum(self.by_gen.values()), 5),
                "per_generation_s": {str(g): round(v, 5) for g, v in sorted(self.by_gen.items())},
                "per_generation_n": {str(g): n for g, n in sorted(self.n_gen.items())},
                "per_region_s": {r: round(v, 5) for r, v in
                                 sorted(self.by_region.items(), key=lambda kv: -kv[1])},
                "objects_collected": self.collected,
                "thresholds": gc.get_threshold(), "enabled": gc.isenabled()}


# =============================================================================================
_PREPARE_PARTIAL = None


def install_regions(reg, state, T, boltz2):
    """Cover `predict_one` completely, so the leftover row is measured and not inferred."""
    import tt_bio.main as M

    ok = {}
    global _PREPARE_PARTIAL
    _PREPARE_PARTIAL = getattr(state, "prepare", None)
    # --- the four top-level steps of the boltz-2 branch of `_WorkerState.predict_one` -------
    ok["prepare"] = reg.patch(state, "prepare", "prepare")
    ok["to_batch"] = reg.patch(M, "to_batch", "to_batch")
    ok["predict_step"] = reg.patch(state.model, "predict_step", "predict_step")
    ok["write_result"] = reg.patch(M, "write_result", "write_result")
    # `state.prepare` is a `partial(prepare_features, ...)` bound at load_model time, so
    # patching `M.prepare_features` here would never be reached. Featurisation is split below
    # the `prepare` region by the stack sampler instead.
    # --- the device phases, so their wall can be subtracted from their parent's -------------
    for cls, label in (("TrunkModule", "trunk"), ("DiffusionModule", "denoise_device"),
                       ("PairformerModule", "pairformer_conf")):
        obj = getattr(T, cls, None)
        if obj is None:
            reg.notes.append(f"no tt_bio.tenstorrent.{cls}")
            continue
        attr = "__call__" if "__call__" in obj.__dict__ else "forward"
        ok[label] = reg.patch(obj, attr, label)
    # --- the sampler: the loop, its per-step host stages, and the confidence head ----------
    ok["sampler"] = reg.patch(boltz2.AtomDiffusion, "sample", "sampler")
    ok["denoiser"] = reg.patch(boltz2.AtomDiffusion, "preconditioned_network_forward", "denoiser")
    ok["align"] = reg.patch(boltz2, "weighted_rigid_align", "align")
    ok["randaug"] = reg.patch(boltz2, "compute_random_augmentation", "randaug")
    ok["digest"] = reg.patch(boltz2, "_write_sample_digest", "digest")
    ok["confidence"] = reg.patch(boltz2.ConfidenceModule, "forward", "confidence")
    # --- the host stages between the trunk and the sampler ----------------------------------
    # `predict_step` exclusive turned out to be the biggest single block of the fold's host
    # path, and `DiffusionConditioning` is 60.7 % of it: 120 GFLOP of dense fp32 matmul with no
    # ttnn implementation at all, running between the trunk and the first denoiser call. Give
    # each of these its own row so the residual table has no glue remainder to hide in.
    ok["rel_pos"] = reg.patch(boltz2.RelativePositionEncoder, "forward", "rel_pos")
    ok["input_embedder"] = reg.patch(boltz2.InputEmbedder, "forward", "input_embedder")
    ok["diffusion_cond"] = reg.patch(boltz2.DiffusionConditioning, "forward", "diffusion_cond")
    ok["pairwise_cond"] = reg.patch(boltz2.PairwiseConditioning, "forward", "pairwise_cond")
    ok["atom_encoder"] = reg.patch(boltz2.AtomEncoder, "forward", "atom_encoder")
    # --- round 2: the rows that were only a sampler share the first time round --------------
    # With TT_BIO_DEVICE_CONDITIONING on, `diffusion_cond` is gone and what is left of the host
    # path is `Boltz2.forward`'s own body plus featurisation plus the writer. Name them.
    for cls, label in (("DistogramModule", "distogram"),
                       ("ContactConditioning", "contact_cond"),
                       ("ConfidenceHeads", "confidence_heads")):
        obj = getattr(boltz2, cls, None)
        if obj is None:
            reg.notes.append(f"no tt_bio.boltz2.{cls}")
            continue
        ok[label] = reg.patch(obj, "forward", label)
    for fn in ("parse_yaml", "parse_a3m", "parse_csv"):
        if hasattr(M, fn):
            ok[fn] = reg.patch(M, fn, fn)
    # `state.prepare` is a partial over `prepare_features`; the tokenizer and the featurizer are
    # bound into it, so reach them through the partial rather than importing a second copy.
    part = _PREPARE_PARTIAL
    bound = list(getattr(part, "args", ()) or ()) + list((getattr(part, "keywords", {}) or {}).values())
    for obj in bound:
        for attr, label in (("tokenize", "tokenize"), ("process", "featurize")):
            if label in ok or obj is None or not hasattr(obj, attr):
                continue
            ok[label] = reg.patch(type(obj), attr, label)
    return ok


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", default="plain,attrib,sample")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--plain-n", type=int, default=2)
    ap.add_argument("--sample-period-ms", type=float, default=1.0)
    ap.add_argument("--ab-env", default=None, help="env var toggled between arms")
    ap.add_argument("--ab-values", default="1,0")
    ap.add_argument("--ab-pairs", type=int, default=3)
    a = ap.parse_args()
    OUT_PATH = a.out
    phases = [p for p in a.phases.split(",") if p]

    import torch
    torch.set_grad_enabled(False)
    import tt_bio.tenstorrent as T
    import tt_bio.boltz2 as boltz2
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"
    msa_dir = HERE / f".msa_{a.size}"

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "loadavg": loadavg(), "size": a.size, "phases": phases,
                  "torch_threads": torch.get_num_threads(),
                  "torch_interop_threads": torch.get_num_interop_threads(),
                  "gc_thresholds": gc.get_threshold(),
                  "instruments": {
                      "brackets": "time.perf_counter wall, no synchronize_device",
                      "sampler": f"{a.sample_period_ms} ms sys._current_frames on the main thread",
                      "gc": "gc.callbacks start/stop timing"}}
    dump()

    one_fold, meta, state = B.build_fold("boltz2", msa_dir, tgt, a3m)
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "load_s", "n_msa", "card_type",
                                            "recycling_steps", "timed_region")
                       if k in meta})
    struct_dir = Path(meta["struct_dir"])
    dump()

    print("=== cold fold (discarded) ===", flush=True)
    t, m = one_fold()
    OUT["cold_s"] = round(t, 3)
    dump()

    def plain(tag):
        t0 = time.perf_counter()
        _t, m = one_fold()
        rec = {"wall_s": round(time.perf_counter() - t0, 4), "plddt": m.get("plddt"),
               "loadavg": loadavg(), "cif": cif_sha(struct_dir)}
        OUT.setdefault("plain", {})[tag] = rec
        print(f"  plain[{tag}] {rec['wall_s']:.3f} s  plddt {rec['plddt']}  "
              f"{list(rec['cif'].values())[0][:16] if rec['cif'] else '-'}", flush=True)
        dump()
        return rec

    if "plain" in phases:
        print("=== plain folds: wall reference, A/A floor, CIF sha256 on THIS card ===", flush=True)
        for i in range(a.plain_n):
            plain(f"a{i}")

    if "attrib" in phases:
        print("=== attrib fold: wall-clock region tree over all of predict_one ===", flush=True)
        reg = Regions()
        ok = install_regions(reg, state, T, boltz2)
        t0 = time.perf_counter()
        _t, m = one_fold()
        wall = time.perf_counter() - t0
        reg.remove()
        tbl = reg.table()
        top = {p: v for p, v in tbl.items() if "/" not in p}
        named_top = sum(v["incl_s"] for v in top.values())
        OUT["attrib"] = {"fold_wall_s": round(wall, 4), "plddt": m.get("plddt"),
                         "loadavg": loadavg(), "patched": ok, "notes": reg.notes,
                         "tree": tbl, "top_level_incl_s": round(named_top, 4),
                         "unattributed_s": round(wall - named_top, 4),
                         "cif": cif_sha(struct_dir)}
        print(f"  fold {wall:.3f} s, top-level regions {named_top:.3f} s, "
              f"unattributed {wall - named_top:.3f} s", flush=True)
        for p, v in sorted(tbl.items(), key=lambda kv: -kv[1]["excl_s"])[:16]:
            print(f"    {p:52s} incl {v['incl_s']:8.3f}  excl {v['excl_s']:8.3f}  "
                  f"n {v['calls']:6d}  {v['ms_per_call']:9.3f} ms/call", flush=True)
        dump()

    if "sample" in phases:
        print("=== sample fold: 1 ms stack sampler + gc timing ===", flush=True)
        reg = Regions()
        ok = install_regions(reg, state, T, boltz2)
        gt = GCTimer()
        gt.on()
        smp = StackSampler(period=a.sample_period_ms * 1e-3)
        smp.start()
        t0 = time.perf_counter()
        _t, m = one_fold()
        wall = time.perf_counter() - t0
        smp.stop.set()
        smp.join(timeout=5)
        gt.off()
        reg.remove()
        tbl = reg.table()
        top = {p: v for p, v in tbl.items() if "/" not in p}
        OUT["sample"] = {"fold_wall_s": round(wall, 4), "plddt": m.get("plddt"),
                         "loadavg": loadavg(), "tree": tbl,
                         "top_level_incl_s": round(sum(v["incl_s"] for v in top.values()), 4),
                         "sampler": smp.report(), "gc": gt.report(), "notes": reg.notes,
                         "cif": cif_sha(struct_dir)}
        print(f"  fold {wall:.3f} s (sampler overhead included), samples {smp.total}, "
              f"gc {gt.report()['total_s']:.3f} s", flush=True)
        for row in OUT["sample"]["sampler"]["by_repo_site"][:20]:
            print(f"    {row['region']:34s} {row['at']:44s} {row['n']:6d} "
                  f"{100*row['share_of_region']:5.1f} %", flush=True)
        print(f"  gc per generation: {gt.report()['per_generation_s']} "
              f"n={gt.report()['per_generation_n']}", flush=True)
        dump()

    if "ab" in phases and a.ab_env:
        v_on, v_off = a.ab_values.split(",")
        print(f"=== ab: {a.ab_env} {v_on} vs {v_off}, interleaved ===", flush=True)
        rows = []
        for i in range(a.ab_pairs):
            for arm, val in (("A", v_on), ("B", v_off)):
                os.environ[a.ab_env] = val
                t0 = time.perf_counter()
                _t, m = one_fold()
                w = time.perf_counter() - t0
                rows.append({"i": i, "arm": arm, "env": val, "wall_s": round(w, 4),
                             "plddt": m.get("plddt"), "cif": cif_sha(struct_dir),
                             "loadavg": loadavg()})
                print(f"  {arm}[{i}] {a.ab_env}={val} {w:.3f} s  "
                      f"{list(rows[-1]['cif'].values())[0][:16]}", flush=True)
                OUT["ab"] = {"env": a.ab_env, "rows": rows}
                dump()
        for arm in ("A", "B"):
            ws = [r["wall_s"] for r in rows if r["arm"] == arm]
            OUT["ab"][f"median_{arm}"] = round(st.median(ws), 4)
        OUT["ab"]["speedup_B_over_A"] = round(OUT["ab"]["median_A"] / OUT["ab"]["median_B"], 4)
        dump()

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
