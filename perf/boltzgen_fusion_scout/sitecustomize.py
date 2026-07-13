"""Spawn-safe instrumentation for the BoltzGen design-forward fusion scout.

Patches ``tt_bio.boltzgen.adapter.load_boltz_checkpoint`` at interpreter startup
so timing also applies inside the per-device subprocess that ``gen run`` spawns.
After the model is built (the Boltz class is imported lazily inside the loader,
during the normal run flow, so we avoid the ``boltzgen`` bare-alias import-order
trap that breaks if you import the model at sitecustomize time), we wrap the
design-forward components with device-synchronized timers:
  - trunk driver (Boltz._tt_trunk_module -> resident TrunkModule __call__)
  - DiffusionConditioning.forward (once per design, host-side PyTorch)
  - AtomDiffusion.sample (the 500-step diffusion loop, shared TTDiffusionModule)
  - Boltz.forward (per-design total)

Writes a per-pid JSON at exit. Design forward = trunk + conditioning + diffusion
sample; affinity/confidence are separate pipeline steps with their own ckpts and
are OFF for the design step.
"""
import atexit
import json
import os
import time

PER_CALL = []
_CUR = []


def _sync(dev):
    try:
        import ttnn
        if dev is not None:
            ttnn.synchronize_device(dev)
    except Exception:
        pass


def _install():
    try:
        import tt_bio.boltzgen.adapter as A
    except Exception as e:
        try:
            with open(os.path.join(os.path.dirname(__file__) or ".",
                      f"prof_noinstall_{os.getpid()}.txt"), "w") as f:
                f.write(f"no install: {e}\n")
        except Exception:
            pass
        return

    devbox = {"d": None}

    def _dev():
        if devbox["d"] is None:
            try:
                import tt_bio.tenstorrent as TT
                devbox["d"] = TT.get_device()
            except Exception:
                devbox["d"] = None
        return devbox["d"]

    def _acc(key, dt):
        if _CUR:
            _CUR[-1][key] = _CUR[-1].get(key, 0.0) + dt

    def _wrap_module_methods(model):
        # whole forward (per-design total)
        orig_fwd = model.forward

        def fwd(*a, **k):
            rec = {}
            _CUR.append(rec)
            d = _dev()
            _sync(d)
            t0 = time.perf_counter()
            r = orig_fwd(*a, **k)
            _sync(d)
            rec["total"] = time.perf_counter() - t0
            _CUR.pop()
            PER_CALL.append(rec)
            return r
        model.forward = fwd

        # trunk: _tt_trunk_module() builds a fresh resident driver each call.
        orig_tm = model._tt_trunk_module

        def tm(*a, **k):
            driver = orig_tm(*a, **k)
            if driver is None or getattr(driver, "_prof_wrapped", False):
                return driver
            orig_call = driver.__call__

            def call(*ca, **ck):
                d = _dev()
                _sync(d)
                t0 = time.perf_counter()
                r = orig_call(*ca, **ck)
                _sync(d)
                _acc("trunk", time.perf_counter() - t0)
                return r
            driver.__call__ = call
            driver._prof_wrapped = True
            return driver
        model._tt_trunk_module = tm

        # DiffusionConditioning.forward (once per design)
        dc = getattr(model, "diffusion_conditioning", None)
        if dc is not None and not getattr(dc, "_prof_wrapped", False):
            orig_dc = dc.forward

            def dc_fwd(*a, **k):
                d = _dev()
                _sync(d)
                t0 = time.perf_counter()
                r = orig_dc(*a, **k)
                _sync(d)
                _acc("conditioning", time.perf_counter() - t0)
                return r
            dc.forward = dc_fwd
            dc._prof_wrapped = True

        # AtomDiffusion.sample (500-step diffusion loop)
        sm = getattr(model, "structure_module", None)
        if sm is not None and not getattr(sm, "_prof_wrapped", False):
            orig_sample = sm.sample

            def sample(*a, **k):
                d = _dev()
                _sync(d)
                t0 = time.perf_counter()
                r = orig_sample(*a, **k)
                _sync(d)
                _acc("diffusion_sample", time.perf_counter() - t0)
                return r
            sm.sample = sample
            sm._prof_wrapped = True

        # optional heads (off for the design step, but instrument if present)
        for name in ("confidence_module", "affinity_module", "affinity_module1",
                     "affinity_module2"):
            m = getattr(model, name, None)
            if m is not None and not getattr(m, "_prof_wrapped", False):
                orig_h = m.forward

                def head_fwd(_orig=orig_h, _key=name, *a, **k):
                    d = _dev()
                    _sync(d)
                    t0 = time.perf_counter()
                    r = _orig(*a, **k)
                    _sync(d)
                    _acc(_key, time.perf_counter() - t0)
                    return r
                m.forward = head_fwd
                m._prof_wrapped = True

    orig_load = A.load_boltz_checkpoint

    def load(*a, **k):
        model = orig_load(*a, **k)
        try:
            _wrap_module_methods(model)
        except Exception as e:
            try:
                with open(os.path.join(os.path.dirname(__file__) or ".",
                          f"prof_wraperr_{os.getpid()}.txt"), "w") as f:
                    f.write(f"wrap err: {e}\n")
            except Exception:
                pass
        return model
    A.load_boltz_checkpoint = load

    # predict.py does `from tt_bio.boltzgen.adapter import load_boltz_checkpoint`;
    # patch the adapter attr BEFORE that import runs (sitecustomize runs first).
    # Also patch the already-bound reference if predict was imported earlier.
    try:
        import tt_bio.boltzgen.task.predict.predict as P
        if getattr(P, "load_boltz_checkpoint", None) is orig_load:
            P.load_boltz_checkpoint = load
    except Exception:
        pass

    @atexit.register
    def _dump():
        if not PER_CALL:
            return
        out = os.path.join(os.path.dirname(__file__) or ".",
                           f"prof_{os.getpid()}.json")
        try:
            with open(out, "w") as f:
                json.dump({"pid": os.getpid(), "calls": PER_CALL}, f, indent=2)
        except Exception:
            pass


_install()
