"""Phase-1 attribution probe for the Boltz-2 affinity leg.

Loaded automatically by every interpreter that has this directory on PYTHONPATH,
including the spawn-ed `tt-bio predict` workers (which is the point: the fold runs
in a child process, so an in-parent monkeypatch would see nothing). Writes one
JSON line per enter/exit to $AFFPROBE_LOG. Inert when that var is unset.

Records the ttnn program-cache entry count at every boundary, which is what makes
the `disable_and_clear_program_cache()` hypothesis testable: a clear shows up as a
count dropping to 0, and the recompile cost shows up as the time the next span
takes while the count climbs back.
"""

import json
import os
import sys
import threading
import time

_LOG = os.environ.get("AFFPROBE_LOG")

if _LOG:
    _T0 = time.monotonic()
    _LK = threading.Lock()
    _SEQ = [0]

    def _emit(**kw):
        kw["t"] = round(time.monotonic() - _T0, 4)
        kw["pid"] = os.getpid()
        try:
            with _LK:
                with open(_LOG, "a") as f:
                    f.write(json.dumps(kw, default=str) + "\n")
                    f.flush()
        except Exception:
            pass

    def _pce():
        """Program-cache entries on the open device, without ever opening one."""
        m = sys.modules.get("tt_bio.tenstorrent")
        dev = getattr(m, "_device", None) if m is not None else None
        if dev is None:
            return None
        try:
            return int(dev.num_program_cache_entries())
        except Exception:
            return -1

    def _wrap(obj, name, label, info=None):
        orig = getattr(obj, name, None)
        if orig is None or getattr(orig, "_affprobe", False):
            return
        def w(self, *a, **kw):
            _SEQ[0] += 1
            i = _SEQ[0]
            ev = {"ev": "in", "op": label, "i": i, "pce": _pce()}
            if info is not None:
                try:
                    ev.update(info(self, a, kw))
                except Exception as e:
                    ev["info_err"] = str(e)
            _emit(**ev)
            s = time.monotonic()
            try:
                return orig(self, *a, **kw)
            finally:
                _emit(ev="out", op=label, i=i, dt=round(time.monotonic() - s, 4), pce=_pce())
        w._affprobe = True
        setattr(obj, name, w)

    def _wrap_func(mod, name, label):
        orig = getattr(mod, name, None)
        if orig is None or getattr(orig, "_affprobe", False):
            return
        def w(*a, **kw):
            _SEQ[0] += 1
            i = _SEQ[0]
            _emit(ev="in", op=label, i=i, pce=_pce())
            s = time.monotonic()
            try:
                return orig(*a, **kw)
            finally:
                _emit(ev="out", op=label, i=i, dt=round(time.monotonic() - s, 4), pce=_pce())
        w._affprobe = True
        setattr(mod, name, w)

    def _fwd_info(self, a, kw):
        d = {"affinity": bool(getattr(self, "affinity_prediction", False)),
             "recycling_steps": kw.get("recycling_steps"),
             "num_sampling_steps": kw.get("num_sampling_steps"),
             "diffusion_samples": kw.get("diffusion_samples"),
             "trunk_fp32": bool(getattr(self, "affinity_trunk_fp32", False))}
        feats = a[0] if a else kw.get("feats")
        try:
            d["n_tok"] = int(feats["token_pad_mask"].shape[-1])
            d["n_atom"] = int(feats["atom_pad_mask"].shape[-1])
        except Exception:
            pass
        return d

    def _sample_info(self, a, kw):
        return {"num_sampling_steps": kw.get("num_sampling_steps"),
                "multiplicity": kw.get("multiplicity"),
                "max_parallel_samples": kw.get("max_parallel_samples"),
                "default_steps": getattr(self, "num_sampling_steps", None)}

    def _patch_boltz2():
        import tt_bio.boltz2 as B
        _wrap(B.Boltz2, "forward", "Boltz2.forward", _fwd_info)
        _wrap(B.AtomDiffusion, "sample", "AtomDiffusion.sample", _sample_info)
        _wrap(B.ConfidenceModule, "forward", "ConfidenceModule")
        _wrap(B.AffinityModule, "forward", "AffinityModule")
        _wrap(B.InputEmbedder, "forward", "InputEmbedder")
        _wrap(B.DiffusionConditioning, "forward", "DiffusionConditioning")
        _wrap(B.DistogramModule, "forward", "DistogramModule")
        _wrap(B.Boltz2, "predict_step", "Boltz2.predict_step")
        cm = getattr(B.Boltz2, "load_from_checkpoint", None)
        if cm is not None and not getattr(cm.__func__, "_affprobe", False):
            orig = cm.__func__
            def w(cls, *a, **kw):
                _SEQ[0] += 1
                i = _SEQ[0]
                _emit(ev="in", op="load_from_checkpoint", i=i, pce=_pce(), ckpt=str(a[0] if a else kw.get("checkpoint_path")))
                s = time.monotonic()
                try:
                    return orig(cls, *a, **kw)
                finally:
                    _emit(ev="out", op="load_from_checkpoint", i=i,
                          dt=round(time.monotonic() - s, 4), pce=_pce())
            w._affprobe = True
            B.Boltz2.load_from_checkpoint = classmethod(w)

    def _patch_tt():
        import tt_bio.tenstorrent as T
        _wrap(T.MSAModule, "forward", "MSAModule")
        _wrap(T.PairformerModule, "forward", "PairformerModule")
        _wrap(T.Fp32PairformerModule, "forward", "Fp32PairformerModule")
        _wrap(T.TrunkModule, "forward", "TrunkModule")
        _wrap(T.TrunkModule, "_iteration", "TrunkModule.iter")
        _wrap(T.DiffusionModule, "forward", "DiffusionModule")
        _wrap_func(T, "cleanup", "tt.cleanup")

    def _patch_worker():
        import tt_bio.worker as W
        _wrap(W._WorkerState, "predict_one", "predict_one")
        _wrap(W._WorkerState, "predict_affinity", "predict_affinity")
        _wrap(W._WorkerState, "load_model", "load_model")
        _wrap(W._WorkerState, "bind_run", "bind_run")

    _TARGETS = (("tt_bio.boltz2", _patch_boltz2),
                ("tt_bio.tenstorrent", _patch_tt),
                ("tt_bio.worker", _patch_worker))
    _DONE = set()

    import builtins
    _real_import = builtins.__import__

    def _hooked(name, *a, **kw):
        m = _real_import(name, *a, **kw)
        for mod, fn in _TARGETS:
            # A module can be in sys.modules while still executing (circular imports),
            # so a failed patch is retried on the next import rather than given up on.
            if mod not in _DONE and mod in sys.modules:
                try:
                    fn()
                except Exception:
                    pass
                else:
                    _DONE.add(mod)
                    _emit(ev="patched", op=mod)
        return m

    builtins.__import__ = _hooked
    _emit(ev="probe_loaded", op="sitecustomize", argv=" ".join(sys.argv[:3]))
