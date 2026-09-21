#!/usr/bin/env python3
"""Boltz-2 by STAGE, on the old tree and on today's tree, so its 1.5006x splits.

`pvx-didittransfer` measured Boltz-2 at 26.770 s then and 17.839 s now, one number for the whole
fold. M16 asks a different question: how much of that 1.5006x is the TRUNK, which every
core-sharing model runs too, and how much is DIFFUSION, which `DiffusionModule` builds for
Boltz-2 and BoltzGen and for nobody else. A whole-fold ratio cannot answer it and a block census
keyed on class names cannot either, because the two trees name their blocks differently in
places. So this times the SEAMS, and the seams are the same five methods in both trees:

    trunk         tt_bio.tenstorrent.TrunkModule.forward        the resident recycling trunk
    diffusion     tt_bio.boltz2.AtomDiffusion.sample            the sampler loop
                  tt_bio.boltz2.DiffusionConditioning.forward   (+ forward_atoms, new tree only)
    confidence    tt_bio.boltz2.ConfidenceHeads.forward
    embed         tt_bio.boltz2.InputEmbedder/DistogramModule/RelativePositionEncoder/BFactorModule

Every target is resolved off the LIVE loaded model, never typed: a name the tree does not have is
recorded in `missing`, so the two arms' tapes are compared as sets rather than assumed equal.

A frame syncs the device on entry and on exit and is charged INCLUSIVE to its stage and SELF to
the stage minus the timed children inside it, so a nested seam is never counted twice. Residual
is fold wall minus the summed self time: host featurisation, the CIF write, and any device work
no seam wraps.

The tape's own charge is measured, not asserted: untaped folds bracket the taped ones in the same
process, which gives both the A/A floor and the taped-against-untaped delta.

AICLK is held for the session and sampled from sysfs at 5 Hz DURING every fold.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, os.environ.get("ALLM_AICLK_DIR", str(ROOT / "perf" / "c14_bfp8")))

STATE = {"dev": None, "on": False, "sync": None}
STACK: list = []
REC: dict = {}
MISSING: list = []
TAPED: list = []


# --------------------------------------------------------------------------- clock
class ClockSampler(threading.Thread):
    """Card AICLK and power from sysfs at 5 Hz. Opens no device, so it is not contention."""

    SYS = Path("/sys/class/tenstorrent")

    def __init__(self, card):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.aiclk: list[int] = []
        self.power: list[int] = []
        self._clk = self.SYS / f"tenstorrent!{card}" / "tt_aiclk"
        if not self._clk.exists():
            cand = next(iter(self.SYS.glob(f"*{card}/tt_aiclk")), None)
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
def _charge(key, dt, child):
    r = REC.setdefault(key, {"n": 0, "incl": 0.0, "self": 0.0})
    r["n"] += 1
    r["incl"] += dt
    r["self"] += dt - child


def wrap(cls, attr, key, keyfn=None):
    orig = cls.__dict__[attr]

    def f(self, *a, **kw):
        if not STATE["on"]:
            return orig(self, *a, **kw)
        k = keyfn(self, key) if keyfn else key
        sync = STATE["sync"]
        sync()
        fr = {"child": 0.0}
        STACK.append(fr)
        t0 = time.perf_counter()
        try:
            return orig(self, *a, **kw)
        finally:
            sync()
            dt = time.perf_counter() - t0
            STACK.pop()
            _charge(k, dt, fr["child"])
            for p in reversed(STACK):
                p["child"] += dt
                break

    f.__wrapped__ = orig
    setattr(cls, attr, f)


_ROLE: dict = {}


def _pa_key(self, key):
    """`PairAssemblyDevice` serves two seams (z-init and diffusion conditioning) from one
    class, so its rows are keyed by first-call order rather than merged into one."""
    r = _ROLE.get(id(self))
    if r is None:
        r = _ROLE[id(self)] = len(_ROLE) + 1
    return f"{key}#{r}"


def install(model, TT):
    """Resolve every seam off the live model and the live tree, and tape what is there."""

    def add(key, owner, attrs, keyfn=None):
        if owner is None:
            MISSING.append(f"{key}: owner absent")
            return
        cls = owner if isinstance(owner, type) else type(owner)
        for attr in attrs:
            if attr in cls.__dict__:
                wrap(cls, attr, key, keyfn)
                TAPED.append(f"{key} -> {cls.__module__}.{cls.__qualname__}.{attr}")
                return
        MISSING.append(f"{key}: {cls.__module__}.{cls.__qualname__} has none of {attrs}")

    add("trunk", getattr(TT, "TrunkModule", None), ("forward", "__call__"))
    add("diffusion_sample", getattr(model, "structure_module", None), ("sample",))
    dc = getattr(model, "diffusion_conditioning", None)
    add("diffusion_cond", dc, ("forward", "__call__"))
    if dc is not None and "forward_atoms" in type(dc).__dict__:
        wrap(type(dc), "forward_atoms", "diffusion_cond_atoms")
        TAPED.append(f"diffusion_cond_atoms -> {type(dc).__module__}."
                     f"{type(dc).__qualname__}.forward_atoms")
    add("confidence", getattr(model, "confidence_module", None), ("forward", "__call__"))
    add("input_embedder", getattr(model, "input_embedder", None), ("forward", "__call__"))
    add("distogram", getattr(model, "distogram_module", None), ("forward", "__call__"))
    add("rel_pos", getattr(model, "rel_pos", None), ("forward", "__call__"))
    add("bfactor", getattr(model, "bfactor_module", None), ("forward", "__call__"))
    pa = getattr(TT, "PairAssemblyDevice", None)
    if pa is not None:
        add("pair_assembly", pa, ("__call__",), keyfn=_pa_key)
    else:
        MISSING.append("pair_assembly: class absent from this tree")


# `trunk` is the shared core; `diffusion*` is the per-model half M16 is about.
STAGES = {
    "trunk": ("trunk",),
    "diffusion": ("diffusion_sample", "diffusion_cond", "diffusion_cond_atoms"),
    "confidence": ("confidence",),
    "embed_and_heads": ("input_embedder", "distogram", "rel_pos", "bfactor",
                        "pair_assembly#1", "pair_assembly#2", "pair_assembly#3"),
}


def snapshot(fold_s):
    seams = {k: {"n": r["n"], "incl_s": round(r["incl"], 6), "self_s": round(r["self"], 6)}
             for k, r in sorted(REC.items())}
    stages = {}
    for stage, keys in STAGES.items():
        s = sum(REC[k]["self"] for k in keys if k in REC)
        if any(k in REC for k in keys):
            stages[stage] = {"self_s": round(s, 6), "share": round(100.0 * s / fold_s, 2)}
    tot = sum(v["self_s"] for v in stages.values())
    stages["residual"] = {"self_s": round(fold_s - tot, 6),
                          "share": round(100.0 * (fold_s - tot) / fold_s, 2)}
    return {"seams": seams, "stages": stages}


def reset():
    REC.clear()
    STACK.clear()
    _ROLE.clear()


def resolve_model(state):
    m = state.model
    for _ in range(5):
        if m is None:
            break
        if hasattr(m, "structure_module"):
            return m
        m = getattr(m, "model", None)
    raise SystemExit("stagesplit: no object with `structure_module` under state.model")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fixdir", type=Path, required=True)
    ap.add_argument("--msadir", type=Path, required=True)
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--folds", default="cold,A,S,B,S,C")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    os.environ.setdefault("TT_BIO_AICLK", str(a.clock))

    import torch  # noqa: F401
    import ttnn
    import tt_bio
    import tt_bio.tenstorrent as TT
    assert Path(tt_bio.__file__).resolve().is_relative_to(ROOT), (
        f"imported tt_bio from {tt_bio.__file__}, not the tree under test ({ROOT})")

    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    # The protocol comes from the TREE UNDER TEST, because doing the model's own work is what
    # is being timed; a protocol imported from the other arm would be a different fold.
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    one_fold, meta, state = B.build_fold(
        a.model, a.msadir, a.fixdir / f"cdk2x2_{a.size}.yaml",
        a.fixdir / f"cdk2x2_{a.size}.a3m")

    dev = TT.get_device()
    STATE["dev"] = dev
    STATE["sync"] = lambda: ttnn.synchronize_device(dev)

    held = 0
    try:
        import aiclk_hold
        held = aiclk_hold.engage(TT.arch_name())
    except Exception as e:                                                     # noqa: BLE001
        meta["aiclk_error"] = repr(e)

    model = resolve_model(state)
    install(model, TT)

    res = {
        "host": socket.gethostname(), "tag": a.tag, "model": a.model, "size": a.size,
        "card": a.card, "tree": str(ROOT),
        "tree_commit": os.environ.get("ALLM_TREE_COMMIT"),
        "aiclk_requested": os.environ.get("TT_BIO_AICLK"), "aiclk_held_mhz": held,
        "env": {k: os.environ.get(k) for k in
                ("TT_VISIBLE_DEVICES", "TT_BIO_LEASE_CARDS", "TT_BIO_LEASE_HOLDER",
                 "TT_MESH_GRAPH_DESC_PATH")},
        "protocol": {"recycling_steps": B.RECYCLING_STEPS,
                     "sampling_steps": B.SAMPLING_STEPS,
                     "diffusion_samples": meta.get("diffusion_samples")},
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        # If the tree took a compiled path the seam is bypassed and the tape is blind, so the
        # flags that decide it are recorded rather than trusted.
        "path_flags": {k: bool(getattr(model, k, False)) for k in
                       ("use_tenstorrent", "is_msa_compiled", "is_pairformer_compiled",
                        "is_template_compiled", "confidence_prediction", "predict_bfactor",
                        "run_trunk_and_structure", "skip_run_structure", "use_templates")},
        "taped": sorted(TAPED), "missing": sorted(MISSING),
        "meta": {k: v for k, v in meta.items() if k not in ("job_cfg",)},
        "folds": [],
    }

    clk = ClockSampler(a.card)
    clk.start()
    for tag in a.folds.split(","):
        tag = tag.strip()
        if not tag:
            continue
        taped = tag.startswith("S")
        reset()
        STATE["on"] = taped
        clk.take()
        t, metrics = one_fold()
        STATE["on"] = False
        row = {"tag": tag, "taped": taped, "fold_s": round(t, 4), "clock": clk.take(),
               "plddt": metrics.get("plddt"), "n_tokens": metrics.get("n_tokens")}
        if taped:
            row.update(snapshot(t))
        res["folds"].append(row)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        st = row.get("stages", {})
        print(f"[{a.tag}/{tag}] {t:.4f} s clk={row['clock'].get('aiclk_median')} "
              f"plddt={row['plddt']} "
              + " ".join(f"{k}={v['self_s']:.3f}" for k, v in st.items()), flush=True)

    clk.stop.set()
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
