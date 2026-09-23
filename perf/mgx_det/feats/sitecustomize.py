"""Device-free capture of what each predict model's host prep hands to its model.

On the PYTHONPATH only when hashseed.py puts it there, with DET_FEATS=<file>. The worker's
load_model is replaced by a stand-in that opens no device, so `tt-bio predict` runs every model's
real featurization (parse, MSA, templates, tokenizer, featurizer) and then calls the stand-in the
way it would call the model: the stand-in hashes every tensor, array and scalar in the call's
arguments, writes one line, and stops the fold. Every device open is replaced by one that raises,
so nothing here can reach a chip, and hashseed.py points the lease dir at its own scratch.
"""
import hashlib
import importlib.abc
import importlib.util
import os
import sys

OUT = os.environ.get("DET_FEATS")


def _h(b):
    return hashlib.sha1(b).hexdigest()[:10]


def _digests(x, name, out, depth=0):
    import numpy as np
    import torch
    if depth > 6:
        return
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            _digests(x[k], f"{name}.{k}", out, depth + 1)
    elif isinstance(x, (list, tuple)):
        if x and all(isinstance(v, (int, float, str, bool)) for v in x):
            out.append(f"{name}={_h(repr(list(x)).encode())}")
        else:
            for i, v in enumerate(x):
                _digests(v, f"{name}[{i}]", out, depth + 1)
    elif isinstance(x, torch.Tensor):
        t = x.detach().cpu().contiguous()
        out.append(f"{name}={_h(f'{t.dtype}{tuple(t.shape)}'.encode() + t.reshape(-1).view(torch.uint8).numpy().tobytes())}")
    elif isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        b = repr(a.tolist()).encode() if a.dtype == object else a.tobytes()
        out.append(f"{name}={_h(f'{a.dtype}{a.shape}'.encode() + b)}")
    elif isinstance(x, (int, float, str, bool, type(None))):
        out.append(f"{name}={_h(repr(x).encode())}")
    elif hasattr(x, "__dict__") and depth < 3:
        _digests(vars(x), name, out, depth + 1)


class _Captured(Exception):
    pass


class _StandIn:
    def __getattr__(self, attr):
        def call(*a, **kw):
            out = []
            _digests(list(a), "arg", out)
            _digests(kw, "kw", out)
            with open(OUT, "a") as fh:
                fh.write(f"{attr} " + " ".join(out) + "\n")
            raise _Captured(f"featurization captured at model.{attr}")
        return call


def _no_device(*a, **kw):
    raise RuntimeError("perf/mgx_det/feats: device open refused")


class _NoDevice:
    """What get_device returns: the worker opens its chip at startup and never touches it here."""
    def __getattr__(self, attr):
        raise RuntimeError(f"perf/mgx_det/feats: device.{attr} used during featurization")


def _load(self, cfg):
    mid = cfg.get("model", "boltz2")
    if mid == "boltz2":                           # the featurizer state the real load builds
        from pathlib import Path
        from tt_bio.data.featurizer import Boltz2Featurizer
        from tt_bio.data.mol import load_canonicals
        from tt_bio.data.tokenize import Boltz2Tokenizer
        self._tokenizer, self._featurizer = Boltz2Tokenizer(), Boltz2Featurizer()
        self._mol_dir = Path(cfg["mol_dir"])
        self._ccd = load_canonicals(self._mol_dir)
    self.model = _StandIn()
    self.config_hash = sys.modules["tt_bio.worker"]._hash_run_config(cfg)
    self.model_id = mid


def _of3_capture(*a, **kw):
    """OpenFold3/OpenBind read weights before the model is called, so capture at their first
    device leg instead, with the fold's host inputs taken from the caller's frame."""
    f = sys._getframe(1).f_locals
    out = []
    for k in ("features", "aux", "msa_feat", "template_feat", "template_slots", "relpos"):
        _digests(f.get(k), k, out)
    with open(OUT, "a") as fh:
        fh.write("run_input_atom_encoder " + " ".join(out) + "\n")
    raise _Captured("featurization captured at run_input_atom_encoder")


PATCH = {
    "tt_bio.openfold3_host_prep": lambda m: setattr(m, "run_input_atom_encoder", _of3_capture),
    "ttnn": lambda m: [setattr(m, f, _no_device) for f in
                       ("open_device", "CreateDevice", "open_mesh_device", "CreateDevices")
                       if hasattr(m, f)],
    "tt_bio.tenstorrent": lambda m: setattr(m, "get_device", lambda *a, **k: _NoDevice()),
    # One worker slot without looking at the chips; its device is never opened (above).
    "tt_bio.runtime": lambda m: setattr(m, "detect_tenstorrent_devices", lambda *a, **k: [0]),
    "tt_bio.worker": lambda m: [setattr(c, "load_model", _load) for c in vars(m).values()
                                if isinstance(c, type) and "load_model" in vars(c)],
}


class _After(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCH:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(mod):
            run(mod)
            PATCH[name](mod)
        spec.loader.exec_module = exec_module
        return spec


if OUT:
    sys.meta_path.insert(0, _After())
