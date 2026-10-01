"""perf/bgx_size/rung.py, plus what `bmm_program_config` narrowed and a digest of every round's
sequence gradient, written beside rung.json.

`autograd.BMM_NARROWED` is empty on every axis whose plans the L1 pricing left alone, so this
is the per-axis evidence that a sweep changed nothing where it should not have. `grad_digest.json`
holds a sha256 per gradient round of the chain gradients and the design loss, so two trees can be
compared bit for bit at the real model.
"""
import atexit, hashlib, json, pathlib, runpy, sys
import numpy as np
import jax
import tt_bio.autograd as ag
from tt_bio import bindcraft2

out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
DIGESTS = []


def _write():
    out.mkdir(parents=True, exist_ok=True)
    nar = getattr(ag, "BMM_NARROWED", {"-": "tree predates the pricing"})
    (out / "narrowed.json").write_text(json.dumps({str(k): v for k, v in nar.items()}, indent=1))
    (out / "grad_digest.json").write_text(json.dumps(DIGESTS, indent=1))


atexit.register(_write)
_design_model_class = bindcraft2.design_model_class


def design_model_class():
    """The class with the digest installed on first use: BindCraft 2 is only importable once
    rung.py has put it on sys.path."""
    cls = _design_model_class()
    if getattr(cls, "_bcw_bmm_digest", False):
        return cls
    real = cls.sequence_gradients

    def sequence_gradients(self, *a, **kw):
        res = real(self, *a, **kw)
        if not kw.get("compile_only"):
            _, grads, loss = res
            h = hashlib.sha256()
            for leaf in jax.tree_util.tree_leaves((grads, loss)):
                h.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
            DIGESTS.append({"round": len(DIGESTS) + 1, "sha256": h.hexdigest(),
                            "loss": float(np.asarray(jax.tree_util.tree_leaves(loss)[0]).ravel()[0])})
        return res

    cls.sequence_gradients = sequence_gradients
    cls._bcw_bmm_digest = True
    return cls


bindcraft2.design_model_class = design_model_class
sys.argv = ["rung.py"] + sys.argv[1:]
runpy.run_path("perf/bgx_size/rung.py", run_name="__main__")
