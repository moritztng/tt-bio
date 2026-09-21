#!/usr/bin/env python3
"""Where a fold actually goes, by block, for any model in tt-bio.

One instrument, every model, because every tt-bio model is built out of the same two base
classes: `tt_bio.tenstorrent.Module` (device blocks, called through `__call__`) and
`TorchWrapper` (torch-side blocks, called through `forward`). So the tape is discovered from
the class tree at run time rather than typed per model, which is the only way a census stays
correct when a model gains a block.

Attribution is a stack, not a label. On entry a timed frame syncs the device, on exit it syncs
again, and the elapsed wall is charged INCLUSIVE to the block and SELF to the block minus the
inclusive wall of the timed children it contains. Op-class labels are not evidence here
(`k10-p1-trimul-critpath`: one `GenericOp` row was seven programs), and neither is a block name
without its children subtracted.

Depth is the knob. `--depth 1` times only the blocks the fold driver calls directly and syncs a
handful of times per fold; `--depth 3` reaches the repeated inner blocks and syncs thousands of
times. The tape's own charge is measured, never assumed: every run folds untaped on both sides of
the taped fold, so the taped total is reported against the untaped one and the untaped pair gives
the A/A floor.

Residual is named, not dropped: fold wall minus the sum of the depth-1 inclusive blocks is host
work (featurisation, the structure write, numpy) plus any device work no block wraps.

AICLK is held at the requested clock for the whole session and sampled from sysfs at 5 Hz DURING
every fold. A number without a during-sampled clock is not a measurement on this part.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "c14_bfp8"))

STATE = {"dev": None, "on": False, "depth": 1, "count_only": False}
STACK: list = []
REC: dict = {}


# --------------------------------------------------------------------------- clock
class ClockSampler(threading.Thread):
    """Card AICLK and power from sysfs at 5 Hz. Opens no device, so it is not contention."""

    ROOT = Path("/sys/class/tenstorrent")

    def __init__(self, card):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.aiclk: list[int] = []
        self.power: list[int] = []
        self._clk = self.ROOT / f"tenstorrent!{card}" / "tt_aiclk"
        if not self._clk.exists():
            cand = next(iter(self.ROOT.glob(f"*{card}/tt_aiclk")), None)
            self._clk = cand if cand else self._clk
        self._pw = next(iter(self._clk.parent.glob("device/hwmon/hwmon*/power1_input")), None)

    def run(self):
        while not self.stop.wait(0.2):
            try:
                v = int(self._clk.read_text().strip())
                if v < 3000:
                    self.aiclk.append(v)
            except (OSError, ValueError):
                pass
            if self._pw is not None:
                try:
                    self.power.append(int(self._pw.read_text().strip()))
                except (OSError, ValueError):
                    pass

    def take(self):
        a, w = sorted(self.aiclk), self.power[:]
        self.aiclk.clear()
        self.power.clear()
        if not a:
            return {"aiclk_n": 0}
        return {"aiclk_min": a[0], "aiclk_max": a[-1], "aiclk_median": a[len(a) // 2],
                "aiclk_mean": round(sum(a) / len(a), 1), "aiclk_n": len(a),
                "power_w_mean": round(sum(w) / len(w) / 1e6, 1) if w else None}


# --------------------------------------------------------------------------- the tape
def discover(TT):
    """Every tt_bio class that defines its own block entry point, with the attribute to wrap.

    Read off the live class tree rather than a list, so a model that grows a block is covered
    without editing this file (`hardcoded-model-list-misses-new-port`).
    """
    out = []
    seen = set()
    for name, mod in list(sys.modules.items()):
        if not (name == "tt_bio" or name.startswith("tt_bio.")) or mod is None:
            continue
        for cname, cls in list(vars(mod).items()):
            if not isinstance(cls, type) or cls.__module__ != name or id(cls) in seen:
                continue
            attrs = []
            if issubclass(cls, TT.TorchWrapper) and "forward" in cls.__dict__:
                attrs.append("forward")
            elif "__call__" in cls.__dict__ and callable(cls.__dict__["__call__"]):
                attrs.append("__call__")
            # A block can be entered by a second, named method instead of `__call__`, and the
            # tape was blind to it: `esmc.SwiGLUFFN.residual` is how every pair transition in
            # the tree is called, so 7.2919 s of ESMFold2 -- 26 % of the fold -- was charged to
            # the caller's self time with no row of its own. Taped under `Class.method` so the
            # two entry points stay distinguishable.
            if "residual" in cls.__dict__ and callable(cls.__dict__["residual"]):
                attrs.append("residual")
            if not attrs:
                continue
            seen.add(id(cls))
            for attr in attrs:
                label = cname if attr in ("forward", "__call__") else f"{cname}.{attr}"
                out.append((name.rsplit(".", 1)[-1], label, cls, attr))
    return out


def install(ttnn, targets):
    def wrap(cls, attr, key):
        orig = cls.__dict__[attr]

        def f(self, *a, **kw):
            if not STATE["on"]:
                return orig(self, *a, **kw)
            if STATE["count_only"]:
                # Membership is COUNTED at run time, at every depth, with no device sync: the
                # question "does this model execute this class at all" needs the count and not
                # the wall, and a sync per frame would price a 1.5 s design out of reach.
                r = REC.setdefault(key, {"n": 0, "incl": 0.0, "self": 0.0,
                                         "depths": defaultdict(int)})
                r["n"] += 1
                return orig(self, *a, **kw)
            if len(STACK) >= STATE["depth"]:
                STACK.append(None)
                try:
                    return orig(self, *a, **kw)
                finally:
                    STACK.pop()
            dev = STATE["dev"]
            d = len(STACK)
            ttnn.synchronize_device(dev)
            fr = {"child": 0.0}
            STACK.append(fr)
            t0 = time.perf_counter()
            try:
                return orig(self, *a, **kw)
            finally:
                ttnn.synchronize_device(dev)
                dt = time.perf_counter() - t0
                STACK.pop()
                r = REC.setdefault(key, {"n": 0, "incl": 0.0, "self": 0.0,
                                         "depths": defaultdict(int)})
                r["n"] += 1
                r["incl"] += dt
                r["self"] += dt - fr["child"]
                r["depths"][d] += 1
                for p in reversed(STACK):
                    if p is not None:
                        p["child"] += dt
                        break

        f.__wrapped__ = orig
        setattr(cls, attr, f)

    for modname, cname, cls, attr in targets:
        wrap(cls, attr, f"{modname}.{cname}")


def snapshot():
    out = {}
    for k, r in REC.items():
        out[k] = {"n": r["n"], "incl_s": round(r["incl"], 6), "self_s": round(r["self"], 6),
                  "depths": {str(d): c for d, c in sorted(r["depths"].items())}}
    return out


def reset():
    REC.clear()
    STACK.clear()


# --------------------------------------------------------------------------- drivers
def driver_predict(a):
    """predict-side models, through the same `_WorkerState.predict_one` the CLI uses."""
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    if a.model == "boltz2":
        # Boltz-2's hyperparameters live inside tt_bio.main's click body, where build_fold cannot
        # reach them, so load_model raises KeyError('conf_kwargs'). The same injection the
        # 26.770 / 17.839 pair and this row's own stage split were measured with, unchanged.
        sys.path.insert(0, str(ROOT / "perf" / "pvx_didit"))
        from cell import patch_boltz2_cfg
        patch_boltz2_cfg()
    tgt = a.target if a.target else ROOT / "perf" / "size512" / "fixtures" / f"cdk2x2_{a.size}.yaml"
    a3m = a.a3m if a.a3m else ROOT / "perf" / "size512" / "fixtures" / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(
        a.model, ROOT / f".msa_allm_{a.model}_{a.size}", Path(tgt), Path(a3m))
    meta["protocol"] = {"recycling_steps": B.RECYCLING_STEPS,
                        "sampling_steps": B.SAMPLING_STEPS}
    return one_fold, meta


def driver_rfd3(a):
    """RFdiffusion3, through `tt_bio.rfd3.design.run_design` -- the same call the CLI makes."""
    import json as _json
    from tt_bio.rfd3 import design as rfd3_design
    weights = Path.home() / ".boltz" / "rfd3" / "weights"
    spec_path = Path(a.target) if a.target else ROOT / "examples" / "rfd3_binder.json"
    specs = _json.loads(spec_path.read_text())
    steps = a.steps or 4
    n = {"i": 0}

    def one():
        n["i"] += 1
        out = Path(f"/tmp/allm_rfd3_{os.getpid()}_{n['i']}")
        t0 = time.perf_counter()
        res = rfd3_design.run_design(specs, out, checkpoint_dir=str(weights), from_pdb=True,
                                     num_timesteps=steps, seed=42, num_designs=1,
                                     batch_size=1, verbose=False)
        return time.perf_counter() - t0, {"plddt": None, "n_designs": len(res)}

    return one, {"spec": str(spec_path), "weights": str(weights),
                 "protocol": {"num_timesteps": steps, "num_designs": 1, "batch_size": 1},
                 "timed_region": "run_design (featurise + sample + CIF write)"}


DRIVERS = {"predict": driver_predict, "rfd3": driver_rfd3}


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--driver", default="predict", choices=sorted(DRIVERS))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--target")
    ap.add_argument("--a3m")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--folds", default="cold,A,B,C")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch  # noqa: F401
    import ttnn
    import tt_bio
    import tt_bio.tenstorrent as TT
    assert Path(tt_bio.__file__).resolve().is_relative_to(ROOT), (
        f"imported tt_bio from {tt_bio.__file__}, not this worktree")

    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    one_fold, meta = DRIVERS[a.driver](a)
    STATE["dev"] = TT.get_device()
    STATE["depth"] = a.depth

    held = 0
    try:
        import aiclk_hold
        held = aiclk_hold.engage(TT.arch_name())
    except Exception as e:                                                     # noqa: BLE001
        meta["aiclk_error"] = repr(e)

    targets = discover(TT)
    install(ttnn, targets)

    clk = ClockSampler(a.card)
    clk.start()

    res = {
        "host": socket.gethostname(), "model": a.model, "size": a.size, "card": a.card,
        "depth": a.depth, "driver": a.driver,
        "git_head": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "aiclk_requested": os.environ.get("TT_BIO_AICLK"), "aiclk_held_mhz": held,
        "taped_classes": len(targets),
        "env": {k: os.environ.get(k) for k in
                ("TT_VISIBLE_DEVICES", "TT_BIO_LEASE_CARDS", "TT_BIO_LEASE_HOLDER")},
        "meta": {k: v for k, v in meta.items() if k not in ("job_cfg",)},
        "folds": [],
    }

    for tag in a.folds.split(","):
        tag = tag.strip()
        if not tag:
            continue
        # `B:2` is a taped fold at depth 2. One session can therefore carry the whole ladder --
        # the coarse split and the split of its dominant block -- with the untaped folds that
        # bracket it giving the A/A floor for both.
        count_only = tag.split(":")[0] == "K"
        taped = tag.split(":")[0] == "B" or count_only
        STATE["count_only"] = count_only
        STATE["depth"] = int(tag.split(":")[1]) if taped and ":" in tag else a.depth
        reset()
        STATE["on"] = taped
        clk.take()
        t, metrics = one_fold()
        STATE["on"] = False
        row = {"tag": tag, "fold_s": round(t, 4), "clock": clk.take(),
               "depth": None if count_only else (STATE["depth"] if taped else None),
               "count_only": count_only,
               "plddt": metrics.get("plddt"), "n_tokens": metrics.get("n_tokens")}
        if taped:
            row["blocks"] = snapshot()
        res["folds"].append(row)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"[{tag}] {t:.4f} s  clock={row['clock'].get('aiclk_median')} MHz "
              f"plddt={row['plddt']}", flush=True)

    clk.stop.set()
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
