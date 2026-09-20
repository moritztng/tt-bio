"""Install the DiffusionModule operand capture in EVERY process of a fold, not just this one.

`tt_bio.main predict` runs the fold in a spawned child, so a patch applied in the parent
never reaches the code that computes the operands -- the first attempt at this capture
recorded nothing and the fold finished clean, which is the quiet version of the failure.
Python imports `sitecustomize` at interpreter start in the parent AND in every spawn
child, so putting the patch here is what makes it reach the process that folds.

Active only when OF3T_CAPTURE_OUT is set, so it is inert on any other run that happens to
have this directory on PYTHONPATH.
"""
import os

_OUT = os.environ.get("OF3T_CAPTURE_OUT")

if _OUT:
    import atexit

    def _install():
        try:
            import torch
            import ttnn
            from tt_bio.openfold3_diffusion_module import OF3DiffusionModule
        except Exception:
            return
        if getattr(OF3DiffusionModule, "_of3t_captured", False):
            return
        orig = OF3DiffusionModule.__call__
        OF3DiffusionModule._of3t_captured = True
        names = orig.__code__.co_varnames[1:orig.__code__.co_argcount]

        def capture(self, *args, **kw):
            if not os.path.exists(_OUT):
                rec = {}
                for n, v in zip(names, args):
                    if isinstance(v, ttnn.Tensor):
                        rec[n] = dict(kind="tensor", t=ttnn.to_torch(v),
                                      dtype=str(v.dtype), layout=str(v.layout))
                    else:
                        rec[n] = dict(kind="scalar", v=v)
                torch.save(rec, _OUT + ".part")
                os.replace(_OUT + ".part", _OUT)
                print("[of3t-capture] %d operands -> %s" % (len(rec), _OUT), flush=True)
            return orig(self, *args, **kw)

        OF3DiffusionModule.__call__ = capture

    # tt_bio is not importable yet at interpreter start, so arm on the first import of it.
    import importlib.abc
    import importlib.machinery
    import sys

    class _Arm(importlib.abc.MetaPathFinder):
        def find_module(self, fullname, path=None):   # pragma: no cover - legacy hook
            return None

        def find_spec(self, fullname, path=None, target=None):
            if fullname == "tt_bio.openfold3_diffusion_module":
                sys.meta_path.remove(self)
                spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
                if spec is not None:
                    orig_exec = spec.loader.exec_module

                    def exec_module(module):
                        orig_exec(module)
                        _install()

                    spec.loader.exec_module = exec_module
                return spec
            return None

    sys.meta_path.insert(0, _Arm())
