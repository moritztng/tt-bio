"""perf/spd/bench.py with every EDM step fingerprinted per sample, to find where two folds of one seed part.

Protenix keeps the diffusion coordinate stream on the host, so a per-sample sha256 of the denoised output and of x at
each of the 200 steps costs nothing on the device. Lines go to $TT_BIO_TRUNK_TAP as `host|<tag>[stepK]|sN|<hash>`,
beside the trunk-exit device taps. Run it in both arms of a comparison.
  TT_BIO_TRUNK_TAP=<file> python perf/spd_bh/steptap.py <bench.py args>
"""
import hashlib
import importlib.abc
import importlib.util
import os
import runpy
import sys


def tap(tag, t, limit=None):
    import torch
    path = os.environ.get("TT_BIO_TRUNK_TAP")
    if not path:
        return
    t = t.detach().to(torch.float32).contiguous()
    with open(path, "a") as fp:
        for i in range(t.shape[0]):
            fp.write(f"host|{tag}|s{i}|{hashlib.sha256(t[i].numpy().tobytes()).hexdigest()[:16]}\n")


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Swap the tap in when bench.py imports tt_bio.protenix, which it does only after the arm's environment is set."""

    def find_spec(self, name, path, target=None):
        if name != "tt_bio.protenix":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(mod):
            run(mod)
            mod.trunk_tap_host = tap
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _PatchOnImport())
os.environ.setdefault("TT_BIO_TRUNK_TAP_CYCLES", "0")
sys.argv = ["perf/spd/bench.py"] + sys.argv[1:]
runpy.run_path("perf/spd/bench.py", run_name="__main__")
