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
#
# `usilu` is the NEGATIVE CONTROL and it narrows nothing. `TT_BIO_UNFUSED_SILU` unfuses an
# activation in the structure module, so it perturbs at the same rounding level as a dtype change
# while touching neither a dtype nor triangle attention. It is here because seed 0 of this fixture
# sits on a sampler basin boundary that ANY bf16-level perturbation flips by ~1.5 A, and an arm
# that flips it is not thereby shown to be inaccurate. Without this control the seed-0 reading of
# every bfp8 region is unattributable.
#
# `widek` was the first choice and is a NULL at 512 aa: measured bit-identical to base, because
# `_tri_att_k_chunks` has a single-entry ladder at every padded length whose divisors do not
# straddle the cap window, 512 included. Kept in the table so nobody spends the folds twice.
#
# `Tmix` is the direct test of the mechanism `Tbias` implicates: region T's bfp8 q/k/v with the
# bias put BACK to bf16 at the SDPA, i.e. the mirror image of `Tbias`. If a mixed dataformat into
# the fused SDPA is what breaks the fold, this breaks too and the operands are not independently
# narrowable; if only `Tbias` breaks, the bias operand alone is at fault.
#
# `DN` is ledger gate 5, the dual-NOC trimul in-projection, narrowed at its DESTINATION only.
# `mm_dualnoc.in_proj` takes the destination format as a positional argument and `a79b2ce55` made
# it allocate in that format instead of a hardcoded bf16, so forcing the argument here narrows the
# region without a production edit. It is scored at 512 aa because at 298 aa the site declines
# every call and is invisible: 2240 of 2240 declined at 298, 560 of 560 SERVED at 512.
# `TDN` is the composition the campaign would actually ship, region T and gate 5 together.
#
# `g`   -> tt_bio.tenstorrent module globals, read at call time
# `env` -> environment variables the code reads live, per call
# `suppress_bias` -> the wrapper below
ARMS = {
    "base":  {},
    # the accumulator arm this row exists for: `z` stored bfloat8_b across the whole Pairformer,
    # every z_update still computed and stored bf16. Five residual `add_`s a layer, 64 layers,
    # 3 recycles, so the quantisation is applied to the accumulation and not to an operand.
    "Z":     {"g": {"_PAIR_Z_B8": True}},
    "T":     {"g": {"_TRIATT_B8": True}},
    "Tbias": {"g": {"_TRIATT_BIAS_B8": True}},
    "Tq":    {"g": {"_TRIATT_B8": True}, "suppress_bias": True},
    "Tmix":  {"g": {"_TRIATT_B8": True}, "bias_bf16": True},
    "DN":    {"dualnoc_b8": True},
    "TDN":   {"g": {"_TRIATT_B8": True}, "dualnoc_b8": True},
    "usilu": {"g": {"_UNFUSED_SILU": True}},
    "widek": {"env": {"TT_BIO_SDPA_WIDE_K": "1"}},
}


def install_dualnoc_arm(DN, ttnn, state):
    """Force `mm_dualnoc.in_proj`'s destination format, and count what the site actually did.

    A dtype arm that never reaches its site reads as a clean null, which is the most expensive
    kind of wrong answer here, so the counters are part of the measurement, not debug output.
    """
    fn = DN.in_proj

    def w(x, weight, ckc, dtype, *a, **kw):
        if state["dualnoc_b8"]:
            state["asked"] += 1
            dtype = ttnn.bfloat8_b
        r = fn(x, weight, ckc, dtype, *a, **kw)
        state["calls"] += 1
        if r is None:
            state["declined"] += 1
        return r

    DN.in_proj = w


def install_bias_suppressor(T, ttnn, state):
    """Let an arm narrow the region's matmul destinations while leaving the bias typecast off.

    `_tri_att_sdpa_inner` reads `(_TRIATT_BIAS_B8 or _TRIATT_B8)` for the bias and nothing else
    dtype-related, and `_tri_att_sdpa` passes it to `_sdpa_masked` by module-global lookup, so
    clearing both flags for the duration of that one call removes the bias narrowing and touches
    nothing else. With `suppress` false this is a pass-through, so the arms that do not ask for it
    are unaffected -- checked by the base digest, which must match the unwrapped runs.
    """
    inner = T._tri_att_sdpa_inner

    def wrapped(q, k, v, bias, *a, **kw):
        if not (state["suppress"] or state["bias_bf16"]):
            return inner(q, k, v, bias, *a, **kw)
        keep = (T._TRIATT_B8, T._TRIATT_BIAS_B8)
        T._TRIATT_B8 = T._TRIATT_BIAS_B8 = False
        back = None
        state["seen"] += 1
        if bias is not None and bias.dtype == ttnn.bfloat8_b:
            state["seen_b8"] += 1
        if state["bias_bf16"] and bias is not None and bias.dtype == ttnn.bfloat8_b:
            state["fired"] += 1
            back = bias = ttnn.typecast(bias, ttnn.bfloat16)
        try:
            return inner(q, k, v, bias, *a, **kw)
        finally:
            T._TRIATT_B8, T._TRIATT_BIAS_B8 = keep
            if back is not None:
                ttnn.deallocate(back)

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
    # would silently run both arms as `base` and score the region as a null. Only the globals the
    # PLANNED arms ask for are required, so the same harness runs against a tree without the flag
    # -- which is itself a control: `Tbias` predates the region and exists on main too.
    want_g, want_env = {}, {}
    for arm in plan:
        for n in ARMS[arm].get("g", {}):
            assert hasattr(T, n), f"tt_bio.tenstorrent has no {n}; arm {arm} is a no-op here"
            want_g[n] = getattr(T, n)
        want_env.update(ARMS[arm].get("env", {}))
    # A pinned value would serve every arm the same way whatever the harness sets.
    for var in list(want_env) + ["TT_BIO_TRIATT_B8", "TT_BIO_TRIATT_BIAS_B8",
                                 "TT_BIO_PAIR_Z_B8", "TT_BIO_PAIR_B8"]:
        assert var not in os.environ, f"{var} may not be pinned; the arm is set per fold"
    assert not any(want_g.values()), f"a region is already on by default: {want_g}"
    defaults = want_g
    # A control that silently does nothing is worse than no control: it reads as a clean null.
    # These counters make the wrapper prove it fired on the arm that asked for it.
    supp = {"suppress": False, "bias_bf16": False, "seen": 0, "seen_b8": 0, "fired": 0,
            "dualnoc_b8": False, "asked": 0, "calls": 0, "declined": 0}
    wrapped = any(ARMS[a].get("suppress_bias") or ARMS[a].get("bias_bf16") for a in plan)
    if wrapped:
        install_bias_suppressor(T, ttnn, supp)
    if any(ARMS[a].get("dualnoc_b8") for a in plan):
        import tt_bio.mm_dualnoc as DN
        install_dualnoc_arm(DN, ttnn, supp)

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

    # KERNEL-PATH. A narrowed `z` is a kernel-eligibility change before it is a byte change:
    # `wk/b2x-bfp8-pair-track` measured `reblock_permute.eligible_back` going 24/24 served to
    # 24/24 declined on a bfloat8_b pair operand, and the fallback moves MORE than the format
    # saves. So every [served, declined] counter on a gate this arm can reach is read per fold,
    # zeroed before it, and a change between the arms is a fallback, which this campaign counts
    # as a failure and not as a cost.
    import importlib
    _MODS = {}
    for _m in ("reblock_permute", "mm_dualnoc", "mm_generic", "triatt_qkv", "triatt_sdpa",
               "trimul_tail", "sdpa_generic", "swiglu_fused", "tenstorrent"):
        try:
            _MODS[_m] = importlib.import_module(f"tt_bio.{_m}")
        except Exception:                                                # noqa: BLE001
            pass

    def _stat_lists():
        for mod_name, mod in _MODS.items():
            for name in dir(mod):
                if not ("STATS" in name or name.endswith("_COUNTS")):
                    continue
                v = getattr(mod, name)
                if isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v):
                    yield f"{mod_name}.{name}", v

    # `_pair_proj_minimal_matmul` has no counter of its own (lever_census wraps it), and it is
    # one of the two gates that refuse a non-bf16 pair operand outright, so it is exactly the one
    # that must be counted here.
    PPMM = [0, 0]
    _ppmm = T._pair_proj_minimal_matmul

    def _pair_proj_minimal_matmul(*a, **kw):
        out = _ppmm(*a, **kw)
        PPMM[0 if out is not None else 1] += 1
        return out

    T._pair_proj_minimal_matmul = _pair_proj_minimal_matmul

    def counters():
        c = {k: list(v) for k, v in _stat_lists() if any(v)}
        c["wrap.PAIR_PROJ_MINIMAL_MATMUL"] = list(PPMM)
        return c

    def zero_counters():
        for _k, v in _stat_lists():
            v[0] = v[1] = 0
        PPMM[0] = PPMM[1] = 0

    def fold(arm, seed, target, keep):
        zero_counters()
        for name, val in defaults.items():
            setattr(T, name, ARMS[arm].get("g", {}).get(name, val))
        for name in want_env:
            os.environ.pop(name, None)
        os.environ.update(ARMS[arm].get("env", {}))
        supp["suppress"] = bool(ARMS[arm].get("suppress_bias", False))
        supp["bias_bf16"] = bool(ARMS[arm].get("bias_bf16", False))
        supp["dualnoc_b8"] = bool(ARMS[arm].get("dualnoc_b8", False))
        supp["seen"] = supp["seen_b8"] = supp["fired"] = 0
        supp["asked"] = supp["calls"] = supp["declined"] = 0
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
                "env_on": {n: os.environ.get(n) for n in want_env},
                "bias_suppressed": supp["suppress"], "bias_forced_bf16": supp["bias_bf16"],
                "sdpa_calls": supp["seen"], "sdpa_calls_bfp8_bias": supp["seen_b8"],
                "bias_downcast_fired": supp["fired"],
                "dualnoc": {"asked_b8": supp["asked"], "calls": supp["calls"],
                            "declined": supp["declined"]},
                "levers": counters(),
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
