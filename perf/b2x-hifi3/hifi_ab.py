#!/usr/bin/env python3
"""Boltz-2 512 aa: trunk math fidelity A/B, end to end, interleaved, in one process.

The claim under test is 1.071x for HiFi3, which was never an end-to-end measurement: it is a
per-op cost multiplied by a TFLOP census over 831.2 of the Pairformer's 872.95 TFLOP. The same
construction has missed in both directions twice in this codebase. This measures the fold.

Everything runs in ONE process (a p300c chip wedges inside ``ttnn.open_device`` on its fourth
open since that chip's last reset), and the arms are INTERLEAVED (A, B, A, B, ...) because
all-A-then-all-B reads a drift as an effect.

The arm is a single in-place write to ``math_fidelity`` on the compute-kernel-config objects the
trunk's modules hold. Two of them, and no more: ``msa_module``'s (which ``TrunkRecycle`` and
``TemplateRecycle`` share by reference) and the 64-block trunk ``pairformer_module``'s. Every
other config object in the model -- diffusion, confidence, affinity, templates -- is a distinct
object and is asserted to be outside the flip set, so an arm provably cannot reach the sampler.

Accuracy is read as Angstrom, not as an RMS on one tensor: each arm writes its CIF, and
``control_rmsd.py`` superposes the arm on the incumbent for the cdk2x2_298 monomeric control AND
for 512 aa, because a fixed-size control cannot certify a shape-dependent change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {}
OUT_PATH: Path | None = None


def dump() -> None:
    if OUT_PATH is not None:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def loadavg() -> list[str]:
    return open("/proc/loadavg").read().split()[:3]


# ---------------------------------------------------------------------------------------------
# the arm: which config objects are the trunk's, and nothing else
# ---------------------------------------------------------------------------------------------
class TrunkFidelity:
    """The trunk's compute-kernel-config objects, selected by identity.

    Selected structurally, not by name: the one ``MSAModule`` and the one 64-block
    ``PairformerModule``. Boltz-2 also builds an 8-block pairformer for confidence and a 2-block
    one for templates; both must stay out, and the assertion below is what proves they did.
    """

    def __init__(self, T, model):
        self.T = T
        self.trunk: dict[int, object] = {}
        self.other: dict[int, str] = {}
        self.rows: list[dict] = []
        self.scan(model)

    def scan(self, model) -> None:
        self.trunk, self.other, self.rows = {}, {}, []
        n64 = 0
        for name, mod in model.named_modules():
            if not isinstance(mod, self.T.TorchWrapper):
                continue
            ckc = getattr(mod, "compute_kernel_config", None)
            if ckc is None:
                continue
            nb = getattr(mod, "n_blocks", None)
            cls = type(mod).__name__
            is_trunk = (cls == "MSAModule") or (cls == "PairformerModule" and nb == 64)
            n64 += cls == "PairformerModule" and nb == 64
            (self.trunk if is_trunk else self.other).__setitem__(
                id(ckc), ckc if is_trunk else f"{name}:{cls}")
            self.rows.append({"name": name, "class": cls, "n_blocks": nb,
                              "ckc_id": id(ckc), "trunk": bool(is_trunk),
                              "fidelity": str(ckc.math_fidelity)})
        assert n64 == 1, f"expected exactly one 64-block trunk pairformer, found {n64}"
        assert any(r["class"] == "MSAModule" for r in self.rows), "no MSAModule found"
        assert not (set(self.trunk) & set(self.other)), \
            "a non-trunk wrapper shares the trunk's config object: the arm would leak"

    def set(self, fid: str) -> list[str]:
        import ttnn
        want = getattr(ttnn.MathFidelity, fid)
        for ckc in self.trunk.values():
            ckc.math_fidelity = want
        got = sorted({str(c.math_fidelity) for c in self.trunk.values()})
        assert got == [f"MathFidelity.{fid}"], f"flip did not take: {got}"
        return got


# ---------------------------------------------------------------------------------------------
# phase walls, with no added sync: both wrappers return torch tensors, so their boundary is
# already synced. Installed for every arm, so it cannot bias one of them.
# ---------------------------------------------------------------------------------------------
class PhaseWall:
    def __init__(self, T):
        self.s: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self._orig = []
        for cls in ("TrunkModule", "DiffusionModule", "PairformerModule", "MSAModule"):
            obj = getattr(T, cls)
            if "forward" not in obj.__dict__:
                continue
            fn = obj.__dict__["forward"]
            self._orig.append((obj, fn))
            setattr(obj, "forward", self._wrap(cls, fn))

    def _wrap(self, name, fn):
        def w(self_obj, *a, **k):
            t = time.perf_counter()
            try:
                return fn(self_obj, *a, **k)
            finally:
                dt = time.perf_counter() - t
                self.s[name] = self.s.get(name, 0.0) + dt
                self.n[name] = self.n.get(name, 0) + 1
        return w

    def take(self) -> dict:
        row = {k: round(v, 4) for k, v in sorted(self.s.items())}
        row["calls"] = dict(sorted(self.n.items()))
        self.s, self.n = {}, {}
        return row

    def remove(self):
        for obj, fn in self._orig:
            setattr(obj, "forward", fn)
        self._orig = []


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--arms", default="HiFi4,HiFi3")
    ap.add_argument("--extra-arms", default="HiFi2",
                    help="measured once each, for the record; not part of the interleave")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--control", type=int, default=298)
    args = ap.parse_args()
    OUT_PATH = args.out
    arms = [a for a in args.arms.split(",") if a]
    extra = [a for a in args.extra_arms.split(",") if a]
    args.cifdir.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn                                                  # noqa: F401
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg                   # injects conf_kwargs
    sys.path[:] = snap
    patch_boltz2_cfg()

    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{args.size}.yaml", fix / f"cdk2x2_{args.size}.a3m"
    msa_dir = Path(__file__).resolve().parent / f".msa_{args.size}"

    import importlib.metadata as im
    OUT["env"] = {
        "host": os.uname().nodename,
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "trunk_math_fidelity_env": os.environ.get("TT_BIO_TRUNK_MATH_FIDELITY"),
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "ttnn": im.version("ttnn"), "torch": torch.__version__,
        "protocol": {"fixture": f"perf/size512/fixtures/cdk2x2_{args.size}.yaml + its a3m",
                     "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
                     "diffusion_samples": B.DIFFUSION_SAMPLES, "seed": B.SEED,
                     "timed_region": "predict_one (featurise + fold + CIF write)",
                     "interleave": arms, "reps": args.reps},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    dump()

    one_fold, meta, state = B.build_fold("boltz2", msa_dir, tgt, a3m)
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "load_s", "n_msa",
                                            "card_type", "aiclk_mhz") if k in meta})
    struct_dir = Path(meta["struct_dir"])
    dump()

    def keep_cifs(tag: str) -> dict:
        got = {}
        for f in sorted(struct_dir.glob("*.cif")):
            dst = args.cifdir / f"{tag}_{f.name}"
            shutil.copyfile(f, dst)
            got[f.name] = sha256_file(dst)
        return got

    wall = PhaseWall(T)

    # The trunk is built lazily on the first forward, so the cold fold comes first and the
    # config objects are re-scanned after it.
    print("=== cold fold (discarded) ===", flush=True)
    cold_s, cold_m = one_fold()
    assert cold_m.get("msa") or meta.get("n_msa"), "fold ran without an MSA"
    OUT["cold"] = {"fold_s": round(cold_s, 3), "plddt": cold_m.get("plddt"),
                   "phase_s": wall.take()}
    print(f"  cold {cold_s:.3f}s plddt={cold_m.get('plddt')}", flush=True)

    fid = TrunkFidelity(T, state.model)
    OUT["ckc_map"] = fid.rows
    OUT["flip_set"] = {"n_objects": len(fid.trunk),
                       "n_excluded_objects": len(fid.other),
                       "excluded": sorted(fid.other.values())}
    print(f"[flip] {len(fid.trunk)} trunk config object(s), "
          f"{len(fid.other)} excluded: {sorted(fid.other.values())}", flush=True)
    dump()

    def fold(arm: str, tag: str, target: Path) -> dict:
        fid.set(arm)
        for p in struct_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        m, _b, _f = state.predict_one(target, dict(meta["job_cfg"]))
        dt = time.perf_counter() - t0
        cifs = keep_cifs(tag)
        rec = {"arm": arm, "tag": tag, "fold_s": round(dt, 3),
               "plddt": m.get("plddt", m.get("complex_plddt")),
               "n_tokens": m.get("n_tokens"), "cifs": cifs,
               "phase_s": wall.take(), "loadavg": loadavg()}
        print(f"  {tag:18s} {arm:6s} {dt:8.3f}s plddt={rec['plddt']} "
              f"trunk={rec['phase_s'].get('TrunkModule')} "
              f"diff={rec['phase_s'].get('DiffusionModule')} "
              f"{list(cifs.values())[0][:16]}", flush=True)
        return rec

    # ---- per-arm warm-up: one fold per arm before any timing ---------------------------------
    print("=== per-arm warm-up (discarded) ===", flush=True)
    OUT["warmup"] = [fold(a, f"warm_{a}", tgt) for a in arms + extra]
    dump()

    # ---- interleaved timed folds -------------------------------------------------------------
    print(f"=== interleaved {'/'.join(arms)}, {args.reps} reps ===", flush=True)
    runs = []
    for i in range(args.reps):
        for a in arms:
            runs.append(fold(a, f"t{i}_{a}", tgt))
            OUT["runs"] = runs
            dump()

    # ---- the extra arms, once each, for the record -------------------------------------------
    for a in extra:
        runs.append(fold(a, f"x_{a}", tgt))
        OUT["runs"] = runs
        dump()

    def arm_rows(a):
        return [r for r in runs if r["arm"] == a and r["tag"].startswith(("t", "x"))]

    summ = {}
    for a in arms + extra:
        ts = [r["fold_s"] for r in arm_rows(a)]
        if not ts:
            continue
        digests = sorted({d for r in arm_rows(a) for d in r["cifs"].values()})
        summ[a] = {
            "n": len(ts), "fold_s": ts, "median_s": round(st.median(ts), 3),
            "min_s": min(ts), "max_s": max(ts),
            "spread_pct": round(100 * (max(ts) - min(ts)) / st.median(ts), 3),
            "trunk_s": round(st.median([r["phase_s"].get("TrunkModule", 0.0)
                                        for r in arm_rows(a)]), 4),
            "diffusion_s": round(st.median([r["phase_s"].get("DiffusionModule", 0.0)
                                            for r in arm_rows(a)]), 4),
            "plddt": sorted({r["plddt"] for r in arm_rows(a)}),
            "cif_sha256": digests, "cif_sha256_16": sorted({d[:16] for d in digests}),
            "bit_identical_within_arm": len(digests) == 1,
        }
    base = arms[0]
    for a, row in summ.items():
        row["speedup_vs_" + base] = round(summ[base]["median_s"] / row["median_s"], 4)
        row["trunk_speedup_vs_" + base] = (
            round(summ[base]["trunk_s"] / row["trunk_s"], 4) if row["trunk_s"] else None)
    summ["aa_floor_pct"] = summ[base]["spread_pct"]
    summ["aa_floor_s"] = round(summ[base]["max_s"] - summ[base]["min_s"], 3)
    OUT["summary"] = summ
    dump()
    print("[summary] " + json.dumps(summ, indent=1), flush=True)

    # ---- the control, per arm ----------------------------------------------------------------
    c_tgt, c_a3m = fix / f"cdk2x2_{args.control}.yaml", fix / f"cdk2x2_{args.control}.a3m"
    try:
        n_msa = B.seed_msa_cache(c_tgt, c_a3m, msa_dir)
        OUT["control"] = {"n_msa": n_msa, "runs": []}
        for a in arms + extra:
            OUT["control"]["runs"].append(fold(a, f"ctl{args.control}_{a}", c_tgt))
            dump()
        B.seed_msa_cache(tgt, a3m, msa_dir)      # restore the 512 MSA
    except Exception:
        import traceback
        OUT["control_error"] = traceback.format_exc()
        print("[control] FAILED\n" + OUT["control_error"], flush=True)
    wall.remove()
    OUT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump()
    print("DONE " + str(OUT_PATH), flush=True)
    T.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
