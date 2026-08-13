#!/usr/bin/env python3
"""E1c -- op census + atom-path share `f` for ONE BoltzGen diffusion step.

The state doc (state/boltzgen-deep-perf.md sec.3) says the 500-step diffusion loop is 53.7 % of
the design forward and that its device call sits ~40-90x away from BOTH the DRAM and the compute
roof, so the binding limit is per-op fixed overhead. Two readings of the op count K prescribe
opposite builds:

  K ~ 1200  ->  39 us/op, ops are individually slow, op count is NOT the lever
  K ~ 4000  ->  12 us/op, classic tiny-op floor, op count IS the only lever

and sec.4's atom-bucket GO/NO-GO needs `f`, the atom encoder+decoder share of the step.

Both are measured here, inside a live design run, on the real cached conditioning:

  1. GRAPH CAPTURE of one whole `_run_diffusion_device` call -> K, the true device program count.
  2. GRAPH CAPTURE of each of the three DiffusionTransformer sections on its own step
     (encoder / token_transformer / decoder) -> the op count split, which answers sec.4's
     second GO clause: does the atom-path op count scale with N_padded/ATOM_WINDOW?
  3. SYNCED per-section timing over a run of steps -> `f` as a share of device time.

Timing and counting are separated on purpose. The synced legs cost a device sync per section and
so inflate the step; they are only ever read as a RATIO. The unsynced step time is measured on
its own steps and is what sec.3's 47.0 ms is compared against.

The run aborts once the census is done -- it never needs the remaining sampling steps.
"""
from __future__ import annotations

import argparse, json, os, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


class Done(Exception):
    """Raised to abort the design run once the census is complete."""


def op_hist(captured):
    """Histogram of device-op names in a captured graph, plus the total count.

    A captured graph is a node list; `function_start` nodes carry the op name in
    params['name']. ttnn python wrappers appear as `ttnn.<op>` and the device ops they lower to
    appear as `ttnn::prim::...` / bare op-struct names, so both views are returned and the
    caller decides which one is `K`.
    """
    py, dev, other = Counter(), Counter(), Counter()
    for n in captured:
        if n.get("node_type") != "function_start":
            continue
        nm = (n.get("params") or {}).get("name", "?")
        if nm.startswith("ttnn.prim.") or "::prim::" in nm:
            dev[nm] += 1
        elif nm.startswith("ttnn."):
            py[nm] += 1
        else:
            other[nm] += 1
    return {
        "n_py": sum(py.values()), "n_dev": sum(dev.values()), "n_other": sum(other.values()),
        "top_py": py.most_common(25), "top_dev": dev.most_common(25),
        "top_other": other.most_common(25),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None)
    ap.add_argument("--protocol", default="protein-anything")
    ap.add_argument("--warm", type=int, default=4, help="steps to skip before measuring")
    ap.add_argument("--time-steps", type=int, default=15, help="steps per timing leg")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    spec = Path(a.spec) if a.spec else here / "binder_fixed100.yaml"
    out_json = Path(a.out) if a.out else here / "opcensus.json"

    import ttnn
    import tt_bio.tenstorrent as T

    R = {"host": __import__("socket").gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"), "spec": str(spec)}
    SEC = {}            # id(DiffusionTransformer instance) -> section name
    CAP = {"want": None}   # section name to graph-capture on this step, or None
    GRAPHS = {}
    TIME = defaultdict(lambda: {"n": 0, "s": 0.0})
    state = {"step": 0, "phase": None, "dev": None}

    # ---- section hook -------------------------------------------------------------------
    _orig_dt_call = T.DiffusionTransformer.__call__

    def dt_call(self, *x, **k):
        sec = SEC.get(id(self))
        if sec is None:                       # not one of the three top-level sections
            return _orig_dt_call(self, *x, **k)
        if CAP["want"] == sec:
            ttnn.synchronize_device(state["dev"])
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                out = _orig_dt_call(self, *x, **k)
                ttnn.synchronize_device(state["dev"])
            finally:
                GRAPHS[sec] = ttnn.graph.end_graph_capture()
            CAP["want"] = None
            return out
        if state["phase"] == "time":
            ttnn.synchronize_device(state["dev"])
            t0 = time.perf_counter()
            out = _orig_dt_call(self, *x, **k)
            ttnn.synchronize_device(state["dev"])
            TIME[sec]["n"] += 1
            TIME[sec]["s"] += time.perf_counter() - t0
            return out
        return _orig_dt_call(self, *x, **k)

    T.DiffusionTransformer.__call__ = dt_call

    # One-shot capture of a single unit (AdaLN / DiffusionTransformerLayer), first call only.
    def unit_hook(cls, name):
        orig = cls.__call__

        def call(self, *x, **k):
            if (CAP.get("unit") != name or name in GRAPHS
                    or getattr(self, "atom_level", False)):
                return orig(self, *x, **k)
            ttnn.synchronize_device(state["dev"])
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                out = orig(self, *x, **k)
                ttnn.synchronize_device(state["dev"])
            finally:
                GRAPHS[name] = ttnn.graph.end_graph_capture()
            return out

        cls.__call__ = call

    unit_hook(T.AdaLN, "AdaLN")
    unit_hook(T.DiffusionTransformerLayer, "DiffusionTransformerLayer")

    # ---- step hook ----------------------------------------------------------------------
    _orig_run = T.DiffusionModule._run_diffusion_device
    _orig_fwd = T.DiffusionModule.forward

    # Split `forward` into staging vs device. The planning pass timed `forward` (47.0 ms,
    # co-tenanted) and called the whole thing "the device call"; `forward` in fact also does
    # `_from_torch(r)`, `_from_torch(times)`, a host pad and `_to_torch(out)[:, :N, :]`. Which
    # side the gap lives on decides whether there is a cheap staging lever at all.
    _orig_from = T.TorchWrapper._from_torch
    _orig_to = T.TorchWrapper._to_torch

    def from_torch(self, *x, **k):
        if state["phase"] != "split":
            return _orig_from(self, *x, **k)
        t0 = time.perf_counter()
        try:
            return _orig_from(self, *x, **k)
        finally:
            TIME["stage:_from_torch"]["n"] += 1
            TIME["stage:_from_torch"]["s"] += time.perf_counter() - t0

    def to_torch(self, *x, **k):
        if state["phase"] != "split":
            return _orig_to(self, *x, **k)
        t0 = time.perf_counter()
        try:
            return _orig_to(self, *x, **k)
        finally:
            TIME["stage:_to_torch"]["n"] += 1
            TIME["stage:_to_torch"]["s"] += time.perf_counter() - t0

    T.TorchWrapper._from_torch = from_torch
    T.TorchWrapper._to_torch = to_torch

    def fwd(self, r, times, s_inputs, s_trunk, *rest, **kw):
        if state["step"] >= a.warm and TIME["forward_total"]["n"] < a.time_steps:
            state["phase"] = "split"
            t0 = time.perf_counter()
            try:
                return _orig_fwd(self, r, times, s_inputs, s_trunk, *rest, **kw)
            finally:
                TIME["forward_total"]["n"] += 1
                TIME["forward_total"]["s"] += time.perf_counter() - t0
                state["phase"] = None
        if not SEC:
            m = self.module
            SEC[id(m.encoder)] = "atom_encoder"
            SEC[id(m.token_transformer)] = "token_transformer"
            SEC[id(m.decoder)] = "atom_decoder"
            state["dev"] = self.tt_device
            R["shapes"] = {
                "MAX_ATOMS_PER_TOKEN": T.MAX_ATOMS_PER_TOKEN,
                "PAIRFORMER_PAD_MULTIPLE": T.PAIRFORMER_PAD_MULTIPLE,
                "ATOM_WINDOW": T.ATOM_WINDOW,
                "ATOM_N_LAYERS": T.ATOM_N_LAYERS, "TOKEN_N_LAYERS": T.TOKEN_N_LAYERS,
                "ATOM_DIM": T.ATOM_DIM, "TOKEN_DIM": T.TOKEN_DIM,
                "N_real": int(r.shape[1]), "seq_len": int(s_inputs.shape[1]),
                "batch": int(r.shape[0]),
            }
        return _orig_fwd(self, r, times, s_inputs, s_trunk, *rest, **kw)

    def run_dev(self, r_dev, times_dev, large_seq_len):
        i = state["step"]
        state["step"] += 1
        W, TS = a.warm, a.time_steps

        # phase 0: warm-up, untouched
        if i < W:
            return _orig_run(self, r_dev, times_dev, large_seq_len)

        # phase 1: unsynced step wall (what sec.3's 47.0 ms is), steps [W, W+TS)
        if W <= i < W + TS:
            # do NOT clear state["phase"] here -- `forward` set it to "split" and `_to_torch`
            # still has to be timed after this returns. The section hook only fires on "time".
            ttnn.synchronize_device(state["dev"])
            t0 = time.perf_counter()
            out = _orig_run(self, r_dev, times_dev, large_seq_len)
            ttnn.synchronize_device(state["dev"])
            TIME["step_unsynced"]["n"] += 1
            TIME["step_unsynced"]["s"] += time.perf_counter() - t0
            return out

        # phase 2: per-section synced timing, steps [W+TS, W+2TS)
        if W + TS <= i < W + 2 * TS:
            state["phase"] = "time"
            ttnn.synchronize_device(state["dev"])
            t0 = time.perf_counter()
            out = _orig_run(self, r_dev, times_dev, large_seq_len)
            ttnn.synchronize_device(state["dev"])
            TIME["step_synced"]["n"] += 1
            TIME["step_synced"]["s"] += time.perf_counter() - t0
            state["phase"] = None
            return out

        # phase 3: one graph capture per step -- whole step, then each section
        j = i - (W + 2 * TS)
        if j == 0:
            ttnn.synchronize_device(state["dev"])
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                out = _orig_run(self, r_dev, times_dev, large_seq_len)
                ttnn.synchronize_device(state["dev"])
            finally:
                GRAPHS["whole_step"] = ttnn.graph.end_graph_capture()
            return out
        if j in (1, 2, 3):
            CAP["want"] = ("atom_encoder", "token_transformer", "atom_decoder")[j - 1]
            out = _orig_run(self, r_dev, times_dev, large_seq_len)
            CAP["want"] = None
            return out
        # L5's census: one AdaLN and one whole DiffusionTransformerLayer, so "192 s-path programs
        # per step" is a measurement and not a read of the source.
        if j in (4, 5):
            CAP["unit"] = ("AdaLN", "DiffusionTransformerLayer")[j - 4]
            out = _orig_run(self, r_dev, times_dev, large_seq_len)
            CAP["unit"] = None
            return out
        raise Done()

    T.DiffusionModule.forward = fwd
    T.DiffusionModule._run_diffusion_device = run_dev

    # ---- run ----------------------------------------------------------------------------
    workdir = Path.home() / "bgdeep_e1c_out"
    if workdir.exists():
        shutil.rmtree(workdir)
    argv = ["run", str(spec), "--output", str(workdir), "--num_designs", "1",
            "--protocol", a.protocol, "--steps", "design",
            "--device_ids", os.environ.get("TT_VISIBLE_DEVICES", "0")]
    from tt_bio.main import _run_boltzgen_cli
    t0 = time.perf_counter()
    aborted = False
    try:
        _run_boltzgen_cli("tt-bio design", argv)
    except Done:
        aborted = True
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    except BaseException as e:                     # the CLI may wrap Done
        if "Done" not in repr(e):
            raise
        aborted = True
    R["wall_s"] = round(time.perf_counter() - t0, 3)
    R["aborted_after_census"] = aborted
    R["steps_seen"] = state["step"]

    # ---- results -------------------------------------------------------------------------
    import importlib.metadata as md
    R["wheel"] = md.version("ttnn")
    R["timing"] = {k: {"calls": v["n"], "total_s": round(v["s"], 4),
                       "ms_per_call": round(1000.0 * v["s"] / max(1, v["n"]), 4)}
                   for k, v in sorted(TIME.items())}
    R["graphs"] = {}
    for k, g in GRAPHS.items():
        try:
            cap = g if isinstance(g, list) else json.loads(g)
        except Exception:
            cap = []
        R["graphs"][k] = op_hist(cap) if cap else {"error": "empty capture", "raw": str(g)[:400]}

    step_ms = R["timing"].get("step_unsynced", {}).get("ms_per_call")
    syn = R["timing"].get("step_synced", {}).get("ms_per_call")
    enc = R["timing"].get("atom_encoder", {}).get("ms_per_call")
    tok = R["timing"].get("token_transformer", {}).get("ms_per_call")
    dec = R["timing"].get("atom_decoder", {}).get("ms_per_call")
    if syn and enc and dec:
        R["f_atom_path"] = round((enc + dec) / syn, 4)
        R["token_share"] = round(tok / syn, 4) if tok else None
        R["sections_sum_share"] = round((enc + tok + dec) / syn, 4) if tok else None
    K = (R["graphs"].get("whole_step") or {}).get("n_dev")
    if K and step_ms:
        R["K_dev_ops"] = K
        R["us_per_dev_op"] = round(1000.0 * step_ms / K, 3)
    Kpy = (R["graphs"].get("whole_step") or {}).get("n_py")
    if Kpy and step_ms:
        R["K_py_ops"] = Kpy
        R["us_per_py_op"] = round(1000.0 * step_ms / Kpy, 3)

    out_json.write_text(json.dumps(R, indent=1))
    print("=== E1C ===")
    print(json.dumps({k: v for k, v in R.items() if k != "graphs"}, indent=1), flush=True)
    for k, v in R["graphs"].items():
        print(f"--- graph {k}: n_dev={v.get('n_dev')} n_py={v.get('n_py')} "
              f"n_other={v.get('n_other')}", flush=True)
        for nm, c in (v.get("top_dev") or [])[:12]:
            print(f"      dev {c:6d}  {nm}", flush=True)
        for nm, c in (v.get("top_py") or [])[:12]:
            print(f"      py  {c:6d}  {nm}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
