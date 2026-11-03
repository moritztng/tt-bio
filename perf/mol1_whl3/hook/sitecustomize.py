"""mol1-whl3: rerun Mol-1's confidence head on the fold's own inputs under lever ablations.

Loaded through PYTHONPATH in every process of a `tt-bio predict` run (the CLI spawns its fold workers), and inert
unless WHL3_DUMP names an output directory. After the fold's own `Protenix._confidences` returns, the hook saves the
head's inputs (s_inputs, s_trunk, z_trunk, coords, the feats the head reads) to WHL3_DUMP/inputs.pt and runs the
head again on them, once per arm:
    served       the active lever set (must reproduce the fold's own numbers)
    host         the same levers, pae/pde/plddt heads on the host (`confidence_samples`)
    none         no levers
    -<lever>     the active set without one lever, for every active lever
Each arm's per-atom pLDDT, mean pLDDT, pTM and the confidence pairformer's single output (s_single) go to
WHL3_DUMP/arms.pt (WHL3_ARMS=served keeps only the first three); `conf_ref.py` grades them against upstream fp32 on the CPU.
"""
import importlib.abc
import importlib.util
import os
import sys

_OUT = os.environ.get("WHL3_DUMP")


def _patch(mod):
    import time

    import torch
    import ttnn

    import tt_bio.tenstorrent as _T

    orig_conf = mod.Protenix._confidences
    orig_bpf = mod.bucketed_pairformer
    taps = []

    def bpf(pf, s, z, dev, *a, **kw):
        so, zo = orig_bpf(pf, s, z, dev, *a, **kw)
        if taps and taps[-1] is not None and pf is taps[-1][0]:
            taps[-1][1].append(ttnn.to_torch(so).float().reshape(-1, so.shape[-1]).clone())
        return so, zo

    mod.bucketed_pairformer = bpf

    # WHL3_LAYERS=1: the trunk's single track after every PairformerLayer up to the first confidence call (layers.pt)
    layers = {"on": bool(os.environ.get("WHL3_LAYERS")), "s": []}
    orig_layer = _T.PairformerLayer.__call__

    # WHL3_SFP32=1: every PairformerLayer carries its single-track residual in fp32 (`s_fp32_residual`); the trunk hands
    # back bf16 as before, so only the residual's precision inside the blocks changes
    sfp32 = bool(os.environ.get("WHL3_SFP32"))
    in_trunk = [False]

    # WHL3_TAP=N: at the N-th PairformerLayer call (0-based, s-carrying), save the single Transition's input, output and
    # every layer norm / linear / product it runs, plus its compute config, to WHL3_DUMP/tap.pt
    tap = {"at": int(os.environ.get("WHL3_TAP", "-1")), "n": 0, "mod": None, "inner": False, "d": {}}
    orig_tr = _T.Transition.__call__

    def grab(name, t):
        tap["d"].setdefault(name, []).append(ttnn.to_torch(t).float().clone())

    def tr_call(self, x, *a, **kw):
        if self is not tap["mod"]:
            return orig_tr(self, x, *a, **kw)
        grab("in", x)
        tap["d"]["ckc"] = repr(self.compute_kernel_config)
        tap["d"]["dtype"] = repr(self.dtype)
        tap["inner"] = True
        try:
            out = orig_tr(self, x, *a, **kw)
        finally:
            tap["inner"] = False
        grab("out", out)
        return out

    _T.Transition.__call__ = tr_call
    for owner, fn in ((ttnn, "layer_norm"), (_T, "_transition_linear"), (ttnn, "multiply_"), (ttnn, "multiply"),
                      (_T, "ws_swiglu")):
        def wrap(*a, _f=getattr(owner, fn), _n=fn, **kw):
            out = _f(*a, **kw)
            if tap["inner"] and isinstance(out, ttnn.Tensor):
                grab(_n + (":" + a[0] if a and isinstance(a[0], str) else ""), out)
            return out
        setattr(owner, fn, wrap)

    def layer_call(self, s, *a, **kw):
        if s is not None and getattr(self, "transition_s", None) is not None:
            if tap["n"] == tap["at"]:
                tap["mod"] = self.transition_s
            tap["n"] += 1
        if sfp32 and in_trunk[0] and s is not None:
            self.s_fp32_residual = True
            if s.dtype != ttnn.float32:
                s = ttnn.typecast(s, ttnn.float32)
        so, zo = orig_layer(self, s, *a, **kw)
        if tap["mod"] is not None and tap["mod"] is getattr(self, "transition_s", None):
            tap["mod"] = None
            if _OUT:
                os.makedirs(_OUT, exist_ok=True)
                torch.save(tap["d"], os.path.join(_OUT, "tap.pt"))
                print("WHL3 tap saved", sorted(tap["d"]), flush=True)
        if layers["on"] and so is not None:
            layers["s"].append(ttnn.to_torch(so).float().reshape(-1, so.shape[-1]).clone())
        return so, zo

    _T.PairformerLayer.__call__ = layer_call

    def run_arm(self, s_inputs, s_trunk, z_trunk, coords, feats, host):
        ch = self.confidence_head
        taps.append((ch.pf, []))
        try:
            if host:
                confs = ch.confidence_samples(s_inputs, s_trunk, z_trunk.result(), coords, feats, True)
            else:
                z_dev = ch.z_base_upload(ch.z_base_host(s_inputs, z_trunk.result()))
                try:
                    confs = ch.confidence_device_samples(s_inputs, s_trunk, z_dev, coords, feats, True)
                finally:
                    ch.drop_device_resident()
        finally:
            _, so = taps.pop()
        return {"plddt": [float(c["plddt"]) for c in confs], "ptm": [float(c["ptm"]) for c in confs],
                "plddt_atom": torch.stack([c["plddt_atom"].float() for c in confs]),
                "plddt_logits": torch.stack([c["plddt_logits"].float() for c in confs]), "s_single": so}

    def _confidences(self, s_inputs, s_trunk, z_trunk, z_base, coords, feats, keep_logits=False):
        if layers["on"]:
            layers["on"] = False
            if _OUT:
                os.makedirs(_OUT, exist_ok=True)
                torch.save(layers.pop("s"), os.path.join(_OUT, "layers.pt"))
        out = orig_conf(self, s_inputs, s_trunk, z_trunk, z_base, coords, feats, keep_logits)
        if not _OUT or os.path.exists(os.path.join(_OUT, "arms.pt")):
            return out
        os.makedirs(_OUT, exist_ok=True)
        keys = ("atom_to_token_idx", "atom_to_tokatom_idx", "distogram_rep_atom_mask", "asym_id", "is_ligand",
                "has_frame", "frame_atom_index", "token_index", "residue_index", "entity_id", "sym_id",
                "is_protein", "is_dna", "is_rna")
        torch.save({"s_inputs": s_inputs.float().cpu(), "s_trunk": s_trunk.float().cpu(),
                    "z_trunk": z_trunk.result().float().cpu(), "coords": torch.stack([c.float() for c in coords]),
                    "feats": {k: feats[k].cpu() for k in keys if k in feats and torch.is_tensor(feats[k])},
                    "levers": sorted(_T._LEVERS), "fold_plddt": [float(c["plddt"]) for c in out]},
                   os.path.join(_OUT, "inputs.pt"))
        active = _T._LEVERS
        arms = [("served", active, False), ("host", active, True), ("none", frozenset(), False)]
        if os.environ.get("WHL3_ARMS") != "served":   # WHL3_ARMS=served: dump the inputs, rerun only the served arm
            arms += [(f"-{lv}", active - {lv}, False) for lv in sorted(active)]
        res = {"fold": {"plddt": [float(c["plddt"]) for c in out]}}
        for name, lv, host in arms:
            t = time.time()
            try:
                with _T.levers(lv):
                    r = run_arm(self, s_inputs, s_trunk, z_trunk, coords, feats, host)
            except Exception as e:  # an arm a lever set cannot run is a finding, not a crash
                r = {"error": f"{type(e).__name__}: {e}"}
            r["s"] = round(time.time() - t, 2)
            res[name] = r
            print(f"WHL3 arm {name}: " + (r.get("error") or
                  "plddt " + " ".join(f"{p:.4f}" for p in r["plddt"]) + f" ptm {r['ptm'][0]:.4f}"), flush=True)
            torch.save(res, os.path.join(_OUT, "arms.part.pt"))
        os.replace(os.path.join(_OUT, "arms.part.pt"), os.path.join(_OUT, "arms.pt"))
        return out

    mod.Protenix._confidences = _confidences

    if sfp32:
        orig_trunk = mod.Trunk.__call__

        def trunk_call(self, *a, **kw):
            in_trunk[0] = True
            try:
                s, z = orig_trunk(self, *a, **kw)
            finally:
                in_trunk[0] = False
            return (ttnn.typecast(s, ttnn.bfloat16) if s.dtype == ttnn.float32 else s), z

        mod.Trunk.__call__ = trunk_call


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name != "tt_bio.protenix":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        self._inner = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch(module)


if _OUT:
    sys.meta_path.insert(0, _Finder())
