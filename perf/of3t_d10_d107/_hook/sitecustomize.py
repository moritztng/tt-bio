"""Both ranking arms off ONE set of diffusion samples, from inside the spawned fold worker.

`tt_bio.main predict` folds in a multiprocessing **spawn** child, so a patch applied in the
launching process never reaches the code that ranks. `PYTHONPATH` is inherited across a spawn
and CPython imports `sitecustomize` at interpreter startup, so this file runs in both.
`perf/of3t_confhead/_hook` is the same idiom and the reason it exists.

A selection-rule change does not move the diffusion samples, so running two folds would put a
0.5 A sampling difference on top of a 4e-03 ranking difference and measure neither. Instead
the wrapper calls `_ptm_iptm` TWICE on the same logits -- once with the frame mask the branch
now passes, once with `has_frame=None`, which is the shipped behaviour -- and records both.
The comparison is then exact by construction: same trunk, same sample, same logits.

It also times `get_token_frame_atoms`, because the mask costs an O(N_atom^2) distance matrix
per sample and a correctness fix that doubles a fold is a different decision.

  OF3T_RECORD   a .jsonl appended to after every confidence call, written as it happens
"""
import builtins
import json
import os
import sys
import time

_REC = os.environ.get("OF3T_RECORD")

if _REC:
    _state = {"installed": False, "frame": None}
    _real_import = builtins.__import__

    def _emit(obj):
        with open(_REC, "a") as fh:
            fh.write(json.dumps(obj) + "\n")

    def _install():
        # Taken from sys.modules, never imported: this runs inside a wrapped `__import__`
        # and importing here re-enters the wrapper.
        au = sys.modules["tt_bio._vendor.openfold3.core.utils.atomize_utils"]
        px = sys.modules["tt_bio.protenix"]

        _real_frames = au.get_token_frame_atoms

        def _timed_frames(batch, x, **kw):
            t0 = time.perf_counter()
            phi, mask = _real_frames(batch=batch, x=x, **kw)
            ms = (time.perf_counter() - t0) * 1e3
            m = mask.bool().reshape(-1)
            atomized = batch["is_atomized"].bool().reshape(-1)
            _state["frame"] = {
                "frame_ms": ms, "n_token": int(m.numel()),
                "n_frameless": int((~m).sum()),
                "n_atomized": int(atomized.sum()),
                "n_frameless_atomized": int((~m & atomized).sum()),
                "n_frameless_standard": int((~m & ~atomized).sum()),
                "n_atom": int(x.shape[-2]),
            }
            return phi, mask

        au.get_token_frame_atoms = _timed_frames

        # A staticmethod accessed through the class is already the plain function on
        # Python 3.10+, so `.__func__` is an AttributeError, not an unwrap.
        _real_ptm = px.ConfidenceHead.__dict__["_ptm_iptm"].__func__

        def _both_arms(pae_logits, asym_id, max_a=32.0, has_frame=None):
            masked = _real_ptm(pae_logits, asym_id, max_a, has_frame)
            shipped = _real_ptm(pae_logits, asym_id, max_a, None)
            row = {"kind": "sample",
                   "ptm_masked": masked[0], "iptm_masked": masked[1],
                   "ptm_shipped": shipped[0], "iptm_shipped": shipped[1],
                   "has_frame_passed": has_frame is not None}
            row.update(_state["frame"] or {})
            _emit(row)
            return masked

        px.ConfidenceHead._ptm_iptm = staticmethod(_both_arms)
        _emit({"kind": "installed", "pid": os.getpid()})

    def _hooked_import(name, *a, **kw):
        mod = _real_import(name, *a, **kw)
        if not _state["installed"]:
            # `in sys.modules` is NOT enough. A module is registered there the moment its
            # execution STARTS, so a hook that fires on presence alone reaches into a
            # half-built `tt_bio.protenix` and gets
            # "partially initialized module ... has no attribute 'ConfidenceHead'".
            # Wait for the attributes themselves, which is the condition that actually
            # matters. Measured: this is what the first openbind run failed on.
            px = sys.modules.get("tt_bio.protenix")
            au = sys.modules.get(
                "tt_bio._vendor.openfold3.core.utils.atomize_utils")
            if (px is not None and au is not None
                    and hasattr(px, "ConfidenceHead")
                    and hasattr(au, "get_token_frame_atoms")):
                _state["installed"] = True      # set BEFORE installing: _install touches
                builtins.__import__ = _real_import   # nothing that would re-enter
                _install()
        return mod

    builtins.__import__ = _hooked_import
