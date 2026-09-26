"""Record a float64 summary of OpenFold3's TRUNK output, so a lever can be graded without the
diffusion sampler in the way.

`TT_BIO_TRIATT_DIVIDING_K` changes triangle attention, which lives in the trunk. Grading it on
the final structure puts the diffusion sampler between the change and the measurement, and on an
unconverged fixture that sampler turns any perturbation into a different structure -- which is
why five folds at 832 gave a 19.6 A seed floor and no resolution. The trunk is deterministic
given its inputs: no seed, no sampler, no floor.

sitecustomize runs before tt_bio exists, and `tt_bio.main` re-execs, so this wraps __import__ and
patches OF3Trunk.__call__ the first time its module appears.
"""
import builtins
import os

_real_import = builtins.__import__
_state = {"patched": False, "n": 0}


def _summarise(obj, out, tag):
    import torch, ttnn
    vals = []
    def walk(o):
        if hasattr(o, "shape") and hasattr(o, "dtype") and "ttnn" in str(type(o)):
            vals.append(o)
        elif isinstance(o, (tuple, list)):
            for e in o:
                walk(e)
    walk(obj)
    rec = {"tag": tag, "n_tensors": len(vals), "tensors": []}
    for i, t in enumerate(vals):
        try:
            h = ttnn.to_torch(t).double()
            # Global scalars fingerprint the whole tensor; the slice is what lets two ARMS be
            # differenced without writing 708 MB of float64 per fold. Deterministic indices, so
            # the same elements are compared every time.
            flat = h.reshape(-1)
            step = max(1, flat.numel() // 4096)
            probe = flat[::step][:4096]
            rec["tensors"].append({"i": i, "shape": list(h.shape),
                                   "rms": float(torch.sqrt(torch.mean(h ** 2))),
                                   "sum": float(h.sum()), "absmax": float(h.abs().max()),
                                   "probe_step": step,
                                   "probe": [float(x) for x in probe]})
        except Exception as e:                                          # noqa: BLE001
            rec["tensors"].append({"i": i, "error": f"{type(e).__name__}"})
    with open(out, "a") as fh:
        import json
        fh.write(json.dumps(rec) + "\n")


def _patch():
    import sys
    mod = sys.modules.get("tt_bio.openfold3_trunk")
    if mod is None or _state["patched"]:
        return
    cls = getattr(mod, "OF3Trunk", None)
    if cls is None:
        return
    out = os.environ.get("TT_TRUNK_DUMP")
    if not out:
        return
    real = cls.__call__

    def wrapped(self, *a, **k):
        r = real(self, *a, **k)
        _state["n"] += 1
        _summarise(r, out, f"call{_state['n']}")
        return r

    cls.__call__ = wrapped
    _state["patched"] = True
    _force_reference_arms()


def _force_reference_arms():
    """Two higher-precision arms, each reachable by rebinding rather than by editing the repo.

    Grading the dividing-k lever needs a reference more accurate than either arm. Neither is
    exposed as an env flag, but both are reachable downstream:

      TT_FORCE_ONE_K_CHUNK      TriangleAttention.tri_att_one_k_chunk is a constructor arg. One
                                k chunk spans the whole key length, so the online softmax makes
                                no running-max rescale and reduces each row in one pass -- the
                                order the torch reference uses. Measured 4.91x smaller row-sum
                                deficit at 320 (tenstorrent.py's own table).
      TT_FORCE_ACCURATE_SOFTMAX _fp32_softmax_attention takes accurate_softmax, and the shipped
                                fall-through passes False.

    They are deliberately run as a PAIR. Each is the better version of one arm's own kernel
    family, so either alone would flatter its own side.
    """
    import os, sys
    T = sys.modules.get("tt_bio.tenstorrent")
    if T is None:
        return
    if os.environ.get("TT_FORCE_ONE_K_CHUNK"):
        ta = getattr(T, "TriangleAttention", None)
        if ta is not None:
            real_init = ta.__init__

            def init(self, *a, **k):
                real_init(self, *a, **k)
                self.tri_att_one_k_chunk = True

            ta.__init__ = init
    if os.environ.get("TT_FORCE_ACCURATE_SOFTMAX"):
        real_fp32 = T._fp32_softmax_attention

        def fp32(*a, **k):
            k["accurate_softmax"] = True
            return real_fp32(*a, **k)

        T._fp32_softmax_attention = fp32


def _imp(name, *a, **k):
    m = _real_import(name, *a, **k)
    if not _state["patched"] and "openfold3_trunk" in name:
        _patch()
    return m


if os.environ.get("TT_TRUNK_DUMP"):
    builtins.__import__ = _imp
