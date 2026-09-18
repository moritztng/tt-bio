#!/usr/bin/env python3
"""Paired same-seed accuracy folds for the bfp8 regions, one process, one device, one model load.

This row edits no production code. An arm is a set of module globals in `tt_bio.tenstorrent`
patched between folds, which works because every one of them is read at CALL time
(`_triatt_dtype()` returns `bfloat8_b if _TRIATT_B8 else _dtype()`), not at import. Running the
arms in one process is what makes the comparison paired: same device, same program cache, same
weights, same MSA, and the seed set on the config rather than on the interpreter.

Arms (see --list). `base` touches nothing and is the reference every other arm is read against.
The last fold repeats the first, and that repeat is the A/A control: if it is not 0.0000 A the
paired read above is not the lever.

Both sizes matter and neither substitutes for the other. cdk2x2_298 is the fixture the 0.35/0.60 A
bar is written against; cdk2x2_512 is the production size and is chimeric, so whole-molecule RMSD
on it reads its unconstrained hinge and the per-pseudo-domain / CA-lDDT readings from
perf/b2z2_fusebias/score.py are what decide there. A 298 aa pass is not evidence at 512 aa:
b2z's transition site scored 0.266 A at 298 aa and moved the 512 aa fold 13.21 A.

AICLK is sampled DURING each fold off the driver's own telemetry, per fold, min/median/max.

    fold_b8.py --out <json> --cifdir <dir> --sizes 512 --plan base:0,1,2,3;T:0,1,2,3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import statistics as st
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import ab_flag_levers as AB  # noqa: E402  -- fixtures, cfg and MSA seeding, unmodified

# arm -> the tt_bio.tenstorrent globals it sets. {} is "touch nothing", i.e. the shipped default.
#
# `T` is the whole region c14-bfp8-fastpath built at 42eebf800: the fused qkv+gate(+bias) pass
# writes its five destinations in bfp8, the fused SDPA destination follows q.dtype, the gate
# multiply_ runs bfp8 x bfp8, and the out projection keeps _dtype() so z_update lands in bf16.
#
# `Tbias` is the pre-existing per-operand flag for the SDPA bias alone, and it is the one operand
# in the region that IS reached through a typecast rather than a matmul destination. Scoring it
# separately is what lets the region's cost be attributed per operand instead of per op.
# `Tq` is `T` with the bias left alone, which is the one decomposition the region supports and
# the only way to attribute its cost per operand. The bias is also the only operand in the region
# reached through an `ttnn.typecast` rather than a matmul destination format, so `T = Tq + Tbias`
# is exactly the "narrow a destination" claim next to the one narrowing that is not.
ARMS = {
    "base":  {},
    "T":     {"_TRIATT_B8": True},
    "Tbias": {"_TRIATT_BIAS_B8": True},
    "Tq":    {"_TRIATT_B8": True, "_suppress_bias_b8": True},
}
GLOBALS = {"_TRIATT_B8", "_TRIATT_BIAS_B8"}          # real tt_bio.tenstorrent names
HARNESS = {"_suppress_bias_b8"}                      # handled by the wrapper below


def install_bias_suppressor(T, state):
    """Let an arm narrow the region's matmul destinations while leaving the bias typecast off.

    `_tri_att_sdpa_inner` reads `(_TRIATT_BIAS_B8 or _TRIATT_B8)` for the bias and nothing else
    dtype-related, and `_tri_att_sdpa` passes it to `_sdpa_masked` by module-global lookup, so
    clearing both flags for the duration of that one call removes the bias narrowing and touches
    nothing else. With `suppress` false this is a pass-through, so the arms that do not ask for it
    are unaffected -- checked by the base digest, which must match the unwrapped runs.
    """
    inner = T._tri_att_sdpa_inner

    def wrapped(*a, **kw):
        if not state["suppress"]:
            return inner(*a, **kw)
        keep = (T._TRIATT_B8, T._TRIATT_BIAS_B8)
        T._TRIATT_B8 = T._TRIATT_BIAS_B8 = False
        try:
            return inner(*a, **kw)
        finally:
            T._TRIATT_B8, T._TRIATT_BIAS_B8 = keep

    T._tri_att_sdpa_inner = wrapped


def aiclk_node(visible: str | None) -> Path | None:
    """The /sys telemetry directory for the card this process opened.

    tt-smi's UMD chip id and /dev/tenstorrent/N are not the same numbering, so resolve through
    the PCI address rather than assuming node == TT_VISIBLE_DEVICES.
    """
    roots = sorted(Path("/sys/class/tenstorrent").glob("tenstorrent!*"))
    if not roots:
        return None
    if visible is None or len(roots) == 1:
        return roots[0]
    want = int(visible.split(",")[0])
    byaddr = sorted(roots, key=lambda r: (r / "device").resolve().name)
    return byaddr[want] if want < len(byaddr) else roots[0]


class Clock:
    """Samples tt_aiclk in a thread so every fold reports the clock it actually ran at."""

    def __init__(self, node: Path | None, period: float = 0.5):
        self.node, self.period, self.samples = node, period, []
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.wait(self.period):
            try:
                self.samples.append(int((self.node / "tt_aiclk").read_text()))
            except Exception:
                pass

    def __enter__(self):
        self.samples = []
        if self.node is not None:
            self._stop.clear()
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)

    def report(self):
        s = self.samples
        if not s:
            return {"n": 0}
        return {"n": len(s), "min": min(s), "median": int(st.median(s)), "max": max(s)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--plan", default="base:0,1,2,3;T:0,1,2,3")
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    ap.add_argument("--list", action="store_true", help="print the arm table and exit")
    args = ap.parse_args()

    if args.list:
        print(json.dumps(ARMS, indent=1))
        return 0

    plan = {}
    for part in args.plan.split(";"):
        arm, _, seeds = part.partition(":")
        assert arm in ARMS, f"unknown arm {arm}; have {sorted(ARMS)}"
        plan[arm] = [int(s) for s in seeds.split(",")]
    sizes = args.sizes.split(",")

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")

    # Refuse to measure a region this checkout does not have: patching a global that is not there
    # would silently run both arms as `base` and score the region as a null.
    for arm in plan:
        for name in ARMS[arm]:
            if name in HARNESS:
                continue
            assert hasattr(T, name), f"tt_bio.tenstorrent has no {name}; arm {arm} is a no-op here"
    # A pinned env value would serve every arm the same way whatever the harness sets.
    for var in ("TT_BIO_TRIATT_B8", "TT_BIO_TRIATT_BIAS_B8"):
        assert var not in os.environ, f"{var} may not be pinned; the arm is set per fold"
    defaults = {n: getattr(T, n) for n in GLOBALS}
    assert not any(defaults.values()), f"a region is already on by default: {defaults}"
    supp = {"suppress": False}
    wrapped = any("_suppress_bias_b8" in ARMS[a] for a in plan)
    if wrapped:
        install_bias_suppressor(T, supp)

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    clk = Clock(aiclk_node(os.environ.get("TT_VISIBLE_DEVICES")))
    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
        "torch": torch.__version__,
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "aiclk_node": str(clk.node),
        "bias_suppressor_installed": wrapped,
        "tt_bio_file": _TB.__file__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(),
        "arms": {a: ARMS[a] for a in plan},
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "plan": args.plan},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()

    work = Path(tempfile.mkdtemp(prefix="b8env-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for size in sizes:
        name = f"cdk2x2_{size}"
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("bfp8-envelope", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    def fold(arm, seed, target, keep):
        for name, val in defaults.items():
            setattr(T, name, ARMS[arm].get(name, val))
        supp["suppress"] = bool(ARMS[arm].get("_suppress_bias_b8", False))
        cfg["seed"] = seed
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        with clk:
            t = time.perf_counter()
            metrics, _b, _f = state.predict_one(target, cfg)
            ttnn.synchronize_device(dev)
            wall = time.perf_counter() - t
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        return {"arm": arm, "seed": seed, "target": target.stem, "fold_s": round(wall, 3),
                "sha256": hashlib.sha256(body).hexdigest()[:16],
                "aiclk": clk.report(),
                "region_on": {n: bool(getattr(T, n)) for n in defaults},
                "bias_suppressed": supp["suppress"],
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}

    first = next(iter(plan))
    for size in sizes:
        target = AB.FIX / f"cdk2x2_{size}.yaml"
        order = []
        for seed in sorted({s for v in plan.values() for s in v}):
            order += [(arm, seed) for arm in plan if seed in plan[arm]]
        order.append((first, plan[first][0]))             # the A/A repeat, last
        seen = set()
        for arm, seed in order:
            tag = f"{arm}-s{seed}" + ("_r1" if (arm, seed) in seen else "")
            seen.add((arm, seed))
            r = fold(arm, seed, target, args.cifdir / f"{size}_{tag}")
            r["tag"] = tag
            out["runs"].append(r)
            c = r["aiclk"]
            print(f"  {size} {tag:12s} {r['fold_s']:7.3f}s sha={r['sha256']} "
                  f"plddt={r['plddt']} aiclk={c.get('median')}MHz "
                  f"[{c.get('min')}-{c.get('max')}]", flush=True)
            dump()

    aa = [r for r in out["runs"] if r["tag"].startswith(f"{first}-s{plan[first][0]}")]
    out["aa_floor_identical"] = {
        size: len({r["sha256"] for r in aa if r["target"].endswith(size)}) == 1 for size in sizes}
    allclk = [s for r in out["runs"] for s in ([r["aiclk"]["min"], r["aiclk"]["max"]]
                                               if r["aiclk"].get("n") else [])]
    out["aiclk_session"] = {"min": min(allclk), "max": max(allclk)} if allclk else {}
    dump()
    print(json.dumps({"aa_floor_identical": out["aa_floor_identical"],
                      "aiclk_session": out["aiclk_session"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
