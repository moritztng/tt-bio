"""Dump OpenFold3's trunk outputs at diffusion entry (bond error by |x| bisect, trunk swap).

Put this directory first on PYTHONPATH and set OF3_TRUNK_DUMP=<dir>. When the sampler is
called, si_trunk, si_input and zij_trunk are read back in fp32 and saved to
<dir>/trunk_<k>.pt. perf/bondref/of3_trunk_swap.py then runs upstream's fp32 diffusion on
them on CPU: flat bond error there puts the TT residual in the diffusion module, growing
error puts it in the trunk.

With OF3_STEP_DUMP=1 as well, every denoising step's network input and output for the first
sample (t, xl_noisy, rl_update, all fp32 on the host) go to <dir>/steps_<k>.pt, so a CPU
replay can run upstream's diffusion module on the same inputs and compare one step at a time.
That mode forces the eager rollout (the traced ones never return to Python per step); the
network is the same, only the EDM combine moves to the host in fp32. For the steps with
t <= OF3_STAGE_TMAX (default 2) it also keeps the stage outputs: the atom encoder's ql, the
DiT input and the DiT output, to name the stage where TT departs from upstream.
"""
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "tt_bio.openfold3_sample_diffusion"
_STAGE_TARGET = "tt_bio.openfold3_diffusion_module"
_DIR = os.environ.get("OF3_TRUNK_DUMP")
_STEPS = os.environ.get("OF3_STEP_DUMP") == "1"
_TMAX = float(os.environ.get("OF3_STAGE_TMAX", "2"))
_STAGE = {}      # stage outputs of the current network call, attached to its step by the EDM hook


def _patch_stages(mod):
    import torch
    import ttnn
    cls = mod.OF3DiffusionModule
    init = cls.__init__
    back = lambda t: torch.Tensor(ttnn.to_torch(t)).float().cpu()

    def wrap(fn, tags):
        def run(*a, **kw):
            if _STAGE.get("on") and tags[0] == "dit_in":
                _STAGE["dit_in"] = back(a[0])[:1].clone()
            y = fn(*a, **kw)
            if _STAGE.get("on"):
                _STAGE[tags[-1]] = back(y)[:1].clone()
            return y
        return run

    def new_init(self, *a, **kw):
        init(self, *a, **kw)
        self.enc_at = wrap(self.enc_at, ("ql_enc",))
        self.dit = wrap(self.dit, ("dit_in", "dit_out"))

    cls.__init__ = new_init


def _patch(mod):
    import torch
    import ttnn
    cls = mod.OF3SampleDiffusion
    orig = cls.__call__
    count = [0]
    steps = []
    edm = cls._edm_host

    def edm_host(xl_noisy, rl_update, atom_mask_host, t, sigma_data):
        if _STEPS:
            st = {"t": float(t), "xl_noisy": xl_noisy[0].detach().clone(),
                  "rl_update": rl_update[0].detach().clone()}
            st.update({k: v for k, v in _STAGE.items() if k != "on"})
            steps.append(st)
            _STAGE.clear()
        return edm(xl_noisy, rl_update, atom_mask_host, t, sigma_data)

    cls._edm_host = staticmethod(edm_host)
    if _STEPS:
        cls._traceable = lambda self, S, width: False
        emb = mod.fourier_noise_emb

        def noise_emb(t, *a, **kw):      # the eager loop calls this once per step, before the network
            _STAGE.clear()
            _STAGE["on"] = float(t) <= _TMAX
            return emb(t, *a, **kw)
        mod.fourier_noise_emb = noise_emb

    def call(self, xl_init_devs, si_trunk_dev, si_input_dev, zij_trunk_dev, *rest, **kw):
        os.makedirs(_DIR, exist_ok=True)
        back = lambda t: torch.Tensor(ttnn.to_torch(t)).float().cpu()
        torch.save({"si_trunk": back(si_trunk_dev), "si_input": back(si_input_dev),
                    "zij_trunk": back(zij_trunk_dev)}, f"{_DIR}/trunk_{count[0]}.pt")
        print(f"[of3_trunk_dump] wrote {_DIR}/trunk_{count[0]}.pt", flush=True)
        steps.clear()
        out = orig(self, xl_init_devs, si_trunk_dev, si_input_dev, zij_trunk_dev, *rest, **kw)
        if steps:
            torch.save(list(steps), f"{_DIR}/steps_{count[0]}.pt")
            print(f"[of3_trunk_dump] wrote {_DIR}/steps_{count[0]}.pt ({len(steps)} steps)", flush=True)
        count[0] += 1
        return out

    cls.__call__ = call


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, target, patch):
        self.target, self.patch = target, patch

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        loader = spec.loader
        exec_module = loader.exec_module

        def run(module):
            exec_module(module)
            self.patch(module)
        loader.exec_module = run
        return spec


if _DIR:
    sys.meta_path.insert(0, _Finder(_TARGET, _patch))
    if _STEPS:
        sys.meta_path.insert(0, _Finder(_STAGE_TARGET, _patch_stages))
