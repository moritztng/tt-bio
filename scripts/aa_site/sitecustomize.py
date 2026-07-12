"""Spawn-safe instrumentation for the Boltz-2 on-device atom attention.

Patches the tenstorrent Diffusion classes at interpreter startup so it also
applies inside mp-spawn worker processes. Within one diffusion sample the 200
sampling steps are warm same-shape repeats, so the per-step MEDIAN is the warm
number (matching the scouts' second-call methodology). Writes JSON per pid.
"""
import atexit
import json
import os
import statistics
import time


def _install():
    try:
        import ttnn
        import tt_bio.tenstorrent as TT
        import tt_bio.boltz2 as B2
    except Exception:
        return

    dev = {"d": None}

    def _sync():
        try:
            if dev["d"] is None:
                dev["d"] = TT.get_device()
            ttnn.synchronize_device(dev["d"])
        except Exception:
            pass

    atom_s, token_s, denoise_s, sample_s = [], [], [], []

    orig_dt = TT.DiffusionTransformer.__call__

    def dt(self, *a, **k):
        # atom-level encoder/decoder pass keys_indexing (4th positional / kwarg);
        # the token transformer does not.
        is_atom = (len(a) >= 4 and a[3] is not None) or (k.get("keys_indexing") is not None)
        _sync(); t0 = time.perf_counter()
        out = orig_dt(self, *a, **k)
        _sync(); dt_s = time.perf_counter() - t0
        (atom_s if is_atom else token_s).append(dt_s)
        return out
    TT.DiffusionTransformer.__call__ = dt

    orig_diff = TT.Diffusion.__call__

    def diff(self, *a, **k):
        _sync(); t0 = time.perf_counter()
        out = orig_diff(self, *a, **k)
        _sync(); denoise_s.append(time.perf_counter() - t0)
        return out
    TT.Diffusion.__call__ = diff

    orig_sample = B2.AtomDiffusion.sample

    def sample(self, *a, **k):
        _sync(); t0 = time.perf_counter()
        out = orig_sample(self, *a, **k)
        _sync(); sample_s.append(time.perf_counter() - t0)
        return out
    B2.AtomDiffusion.sample = sample

    def dump():
        if not denoise_s:
            return
        # steps within a sample are warm repeats after the first ~2; use medians
        def med(x):
            return statistics.median(x) if x else 0.0
        n_steps = len(denoise_s)
        atom_per_step = 2 * med(atom_s) if atom_s else 0.0  # enc + dec per step
        token_per_step = med(token_s)
        denoise_per_step = med(denoise_s)
        out = {
            "pid": os.getpid(),
            "atom_level_calls": len(atom_s),
            "token_calls": len(token_s),
            "denoise_calls": n_steps,
            "atom_median_call_s": med(atom_s),
            "atom_per_step_s": atom_per_step,
            "token_per_step_s": token_per_step,
            "denoise_per_step_s": denoise_per_step,
            "atom_share_of_denoise": (atom_per_step / denoise_per_step) if denoise_per_step else 0.0,
            "sample_walls_s": sample_s,
        }
        path = os.environ.get("AA_BOLTZ_OUT", "/tmp/aa_boltz_timing")
        with open(f"{path}.{os.getpid()}.json", "w") as f:
            json.dump(out, f)

    atexit.register(dump)


_install()
