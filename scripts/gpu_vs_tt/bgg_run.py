"""In-process BoltzGen runner for the GPU benchmark: per-design wall, per-step wall, kernel counters.

Runs the shipped `boltzgen run` CLI in-process (`--no_subprocess` is mandatory, otherwise the
pipeline forks the design step and none of the patches below apply in the child) with four probes
attached. Nothing about the model, the config or the flags is changed; every patch either stamps a
timestamp or increments a counter.

1. `Boltz.predict_step` is wrapped in `torch.cuda.synchronize()` on both sides. One predict_step is
   one design at `diffusion_batch_size=1`: trunk (3 recycles) + the whole denoising loop + the
   coordinate post-processing. Featurization runs in the dataloader worker and weight load happens
   once before, so both sit outside the timed region. An unsynced region lets the next host call
   absorb device time and has inverted a ranking before, so the sync is not decoration.
2. `diffusion.optionally_tqdm` -- the single call site is the denoising loop (diffusion.py:567) --
   is replaced with a stamping generator, giving one timestamp per diffusion step. This is the same
   quantity the Tenstorrent ladder reads off its `diff k/N` lines.
3. `Boltz.load_checkpoint_weights` is stamped. The shipped default is TWO design checkpoints
   (`boltzgen1_diverse.ckpt` + `boltzgen1_adherence.ckpt`, half the designs each), and the switch is
   a full `torch.load` + `load_state_dict` executed INSIDE a predict_step. The design that pays it
   is not a warm design and the driver drops it.
4. The cuEquivariance kernel entry points and their torch fallbacks are counted, so "the fast path
   is engaged" is a number and not a flag that was passed.

Everything is printed as one-line records on stdout for the driver to reduce:
    ENV {json}          once, before the run
    DESIGN <idx> <t0> <t1>
    STEP <t>
    CKPTSWITCH <t> <path>
    COUNTERS {json}     once, after the run
"""

import atexit
import json
import sys
import time

import torch

import boltzgen.model.layers.triangular as TRI
import boltzgen.model.layers.triangular_attention.primitives as PRIM
import boltzgen.model.models.boltz as BOLTZ
import boltzgen.model.modules.diffusion as DIFF

C = {
    "cueq_trimul": 0,          # cuEquivariance triangle_multiplicative_update
    "cueq_triatt": 0,          # cuEquivariance triangle_attention
    "trimul_forward_total": 0,  # every triangular-multiply forward, kernel or not
    "triatt_torch_fallback": 0,  # the torch matmul+softmax path; must stay 0
    "torch_sdpa": 0,           # torch SDPA, which is where the diffusion token attention goes
    "ckpt_switches": 0,
}


def _count_kernel_trimul(*a, **kw):
    C["cueq_trimul"] += 1
    return _orig_trimul(*a, **kw)


def _count_kernel_triatt(*a, **kw):
    C["cueq_triatt"] += 1
    return _orig_triatt(*a, **kw)


def _count_torch_triatt(*a, **kw):
    C["triatt_torch_fallback"] += 1
    return _orig_torch_attn(*a, **kw)


def _count_sdpa(*a, **kw):
    C["torch_sdpa"] += 1
    return _orig_sdpa(*a, **kw)


_orig_trimul = TRI._kernel_triangular_mult
_orig_triatt = PRIM.kernel_triangular_attn
_orig_torch_attn = PRIM._attention
_orig_sdpa = torch.nn.functional.scaled_dot_product_attention

TRI._kernel_triangular_mult = torch.compiler.disable(_count_kernel_trimul)
PRIM.kernel_triangular_attn = torch.compiler.disable(_count_kernel_triatt)
PRIM._attention = _count_torch_triatt
torch.nn.functional.scaled_dot_product_attention = _count_sdpa

# Total triangular-multiply forwards, so a silent fallback shows up as
# trimul_forward_total > cueq_trimul instead of hiding.
for _cls in (TRI.MiniTriangularUpdate, TRI.TriangleMultiplicationOutgoing,
             TRI.TriangleMultiplicationIncoming):
    def _wrap(cls):
        _f = cls.forward

        def forward(self, *a, **kw):
            C["trimul_forward_total"] += 1
            return _f(self, *a, **kw)
        cls.forward = forward
    _wrap(_cls)


def _stamping(iterable, use_tqdm=True, desc=None, **kw):
    for item in iterable:
        sys.stdout.write("STEP %.6f\n" % time.time())
        sys.stdout.flush()
        yield item


DIFF.optionally_tqdm = _stamping

_orig_predict_step = BOLTZ.Boltz.predict_step


def _timed_predict_step(self, batch, batch_idx, *a, **kw):
    torch.cuda.synchronize()
    t0 = time.time()
    out = _orig_predict_step(self, batch, batch_idx, *a, **kw)
    torch.cuda.synchronize()
    t1 = time.time()
    sys.stdout.write("DESIGN %d %.6f %.6f\n" % (batch_idx, t0, t1))
    sys.stdout.flush()
    return out


BOLTZ.Boltz.predict_step = _timed_predict_step

_orig_load_ckpt = BOLTZ.Boltz.load_checkpoint_weights


def _stamped_load_ckpt(self, checkpoint_path):
    C["ckpt_switches"] += 1
    sys.stdout.write("CKPTSWITCH %.6f %s\n" % (time.time(), checkpoint_path))
    sys.stdout.flush()
    return _orig_load_ckpt(self, checkpoint_path)


BOLTZ.Boltz.load_checkpoint_weights = _stamped_load_ckpt


def _dump_counters():
    sys.stdout.write("COUNTERS " + json.dumps(C) + "\n")
    sys.stdout.flush()


atexit.register(_dump_counters)


def _version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception as exc:                                  # noqa: BLE001
        return "unavailable: %s" % exc


env = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "boltzgen": _version("boltzgen"),
    "cuequivariance_torch": _version("cuequivariance-torch"),
    "cuequivariance_ops_torch_cu12": _version("cuequivariance-ops-torch-cu12"),
    "cuequivariance_ops_cu12": _version("cuequivariance-ops-cu12"),
    "argv": sys.argv[1:],
}
sys.stdout.write("ENV " + json.dumps(env) + "\n")
sys.stdout.flush()

from boltzgen.cli.boltzgen import main  # noqa: E402

sys.argv = ["boltzgen"] + sys.argv[1:]
main()
