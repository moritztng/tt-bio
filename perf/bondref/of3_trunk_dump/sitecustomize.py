"""Dump OpenFold3's trunk outputs at diffusion entry (bond error by |x| bisect, trunk swap).

Put this directory first on PYTHONPATH and set OF3_TRUNK_DUMP=<dir>. When the sampler is
called, si_trunk, si_input and zij_trunk are read back in fp32 and saved to
<dir>/trunk_<k>.pt. perf/bondref/of3_trunk_swap.py then runs upstream's fp32 diffusion on
them on CPU: flat bond error there puts the TT residual in the diffusion module, growing
error puts it in the trunk.
"""
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "tt_bio.openfold3_sample_diffusion"
_DIR = os.environ.get("OF3_TRUNK_DUMP")


def _patch(mod):
    import torch
    import ttnn
    cls = mod.OF3SampleDiffusion
    orig = cls.__call__
    count = [0]

    def call(self, xl_init_devs, si_trunk_dev, si_input_dev, zij_trunk_dev, *rest, **kw):
        os.makedirs(_DIR, exist_ok=True)
        back = lambda t: torch.Tensor(ttnn.to_torch(t)).float().cpu()
        torch.save({"si_trunk": back(si_trunk_dev), "si_input": back(si_input_dev),
                    "zij_trunk": back(zij_trunk_dev)}, f"{_DIR}/trunk_{count[0]}.pt")
        print(f"[of3_trunk_dump] wrote {_DIR}/trunk_{count[0]}.pt", flush=True)
        count[0] += 1
        return orig(self, xl_init_devs, si_trunk_dev, si_input_dev, zij_trunk_dev, *rest, **kw)

    cls.__call__ = call


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        loader = spec.loader
        exec_module = loader.exec_module

        def run(module):
            exec_module(module)
            _patch(module)
        loader.exec_module = run
        return spec


if _DIR:
    sys.meta_path.insert(0, _Finder())
