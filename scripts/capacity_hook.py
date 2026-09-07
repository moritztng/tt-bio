"""In-process hook for the capacity gate's two tiers. Loaded by every interpreter in the run.

`tt-bio predict` folds in a SPAWNED worker process, not the launcher, so a patch installed by
the launcher would not reach the process that opens the device. The gate therefore writes a
`sitecustomize.py` into a scratch directory and prepends it to PYTHONPATH: CPython imports
`sitecustomize` in every interpreter it starts, parent and spawned child alike, so the hook
lands wherever the fold actually happens. `install()` is a no-op unless TT_BIO_CAPACITY_HOOK
names a mode, which keeps the scratch dir harmless if it is ever left on a PYTHONPATH.

Two modes, one mechanism. Both need the same thing -- the block lists a model builds -- so both
are driven off one post-`__init__` walk:

  Both modes also emit a per-block-call HEARTBEAT into TT_BIO_CAPACITY_HOOK_BEAT. The CLI's own
  progress stream is per recycle, which at 1504 tokens is coarse enough that a legitimately slow
  block looks exactly like a hang: OpenFold3 spent over five minutes inside ONE trunk block's
  triangle attention here, computing the whole time. A stall detector reading only the coarse
  signal would have called that a STALL, which is the same lie as a green gate in the other
  direction. The heartbeat is a byte appended per block call, so forward progress INSIDE a stage
  is visible.

  screen     truncate every block list to its first element. Class A failures (one oversized
             shape-determined tensor) appear on the FIRST execution of the op, so one block of
             a homogeneous stack surfaces them at a fraction of the wall-clock. This mode can
             only ever make the gate FASTER or less sensitive, never more, which is why a
             screen result is allowed to FAIL a model and is never allowed to PASS one.

  residency  leave the depth alone and sample the DRAM allocator after each block call, to get
             a real high-water mark instead of a final-state reading. Sampling is in-thread at
             a block boundary (the allocator is host-side bookkeeping and reading it drains the
             pipeline, so a polling thread would be both unsafe and a perf lie).

Every mode records WHAT IT TOUCHED into TT_BIO_CAPACITY_HOOK_OUT. A hook that silently matched
nothing would make `screen` a full-depth run reported as a screen, and would make `residency`
report a peak DRAM figure it never measured. The gate reads that file and says "no truncation
applied" or "peak unmeasured" rather than inventing either.
"""

from __future__ import annotations

import atexit
import json
import os
import sys

# Wrapping every class in these would patch the torch reference implementations and vendored
# upstream trees, neither of which is on the device path a capacity run exercises.
_SKIP_PREFIXES = ("tt_bio._vendor", "tt_bio.reference", "tt_bio.af2_reference")

#: Attribute names that hold a homogeneous stack of repeated blocks. Both spellings are used
#: across the ports (`self.blocks = [...]` in the ttnn models, `self.layers` in the
#: torch-derived ones); nothing else in the tree names a list this way.
_STACK_ATTRS = ("blocks", "layers")

#: Sampling is cheap but not free (~6 us per `get_memory_view`). A stack of 48 blocks over 10
#: recycles is 480 samples, which is noise; an inner per-token block could be six figures. So
#: the first _SAMPLE_DENSE samples are taken on every call and the rest every 16th.
_SAMPLE_DENSE = 2000
_SAMPLE_STRIDE = 16

_beat_path = None
_beat_fh = None

_state = {
    "mode": None,
    "truncated": [],      # [[qualname, attr, original_len]] -- proof the screen applied
    "instrumented": [],   # [qualname] -- proof a peak was actually sampled
    "dram_peak_bytes": None,
    "dram_total_bytes": None,
    "dram_banks": None,
    "dram_largest_free_at_peak": None,
    "samples": 0,
    "pid": os.getpid(),
    "errors": [],
}


def _record(err: str) -> None:
    if err not in _state["errors"]:
        _state["errors"].append(err)


def _flush() -> None:
    """Write this process's findings. Every interpreter writes its own file: the launcher patches
    nothing and folds nothing, so a single shared path would let its empty result overwrite the
    worker's real one."""
    out = os.environ.get("TT_BIO_CAPACITY_HOOK_OUT")
    if not out:
        return
    try:
        with open(f"{out}.{os.getpid()}.json", "w") as fp:
            json.dump(_state, fp, indent=1, sort_keys=True)
    except OSError:
        pass


def _beat() -> None:
    """One byte per block call: forward progress inside a stage, for the stall detector."""
    global _beat_fh
    if _beat_path is None:
        return
    try:
        if _beat_fh is None:
            _beat_fh = open(_beat_path, "ab", buffering=0)
        _beat_fh.write(b".")
    except OSError:
        pass


def _sample_dram() -> None:
    """DRAM allocator high-water, sampled at a block boundary in the calling thread."""
    _beat()
    _state["samples"] += 1
    n = _state["samples"]
    if n > _SAMPLE_DENSE and n % _SAMPLE_STRIDE:
        return
    if _state["mode"] != "residency":
        return                                # screen mode wants the heartbeat, not the sampling
    try:
        import ttnn
        from tt_bio.tenstorrent import get_device
        dev = get_device()
        if dev is None:
            return
        v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        banks = int(v.num_banks)
        used = int(v.total_bytes_allocated_per_bank) * banks
        _state["dram_banks"] = banks
        _state["dram_total_bytes"] = int(v.total_bytes_per_bank) * banks
        if used > (_state["dram_peak_bytes"] or -1):
            _state["dram_peak_bytes"] = used
            _state["dram_largest_free_at_peak"] = int(v.largest_contiguous_bytes_free_per_bank)
    except Exception as exc:                      # never let instrumentation fail a fold
        _record(f"dram sample: {type(exc).__name__}: {exc}")


def _instrument_class(cls) -> None:
    """Sample DRAM after each call of `cls`. Only a method the class defines ITSELF is wrapped:
    reaching an inherited `nn.Module.__call__` would instrument every torch module in the
    process, including the ones the fold does not run."""
    for name in ("__call__", "forward"):
        fn = cls.__dict__.get(name)
        if fn is None or getattr(fn, "_capacity_wrapped", False):
            continue

        def wrapper(*a, __fn=fn, **kw):
            r = __fn(*a, **kw)
            _sample_dram()
            return r

        wrapper._capacity_wrapped = True
        try:
            setattr(cls, name, wrapper)
        except (AttributeError, TypeError) as exc:
            _record(f"instrument {cls.__qualname__}.{name}: {exc}")
            continue
        _state["instrumented"].append(f"{cls.__module__}.{cls.__qualname__}.{name}")
        return


def _visit(obj) -> None:
    """Post-`__init__` walk: find this object's block stacks and act on them."""
    mode = _state["mode"]
    if mode is None:
        return                                # disarmed: never touch a model

    for attr in _STACK_ATTRS:
        try:
            stack = getattr(obj, attr, None)
        except Exception:
            continue
        # A plain list in the ttnn ports, an nn.ModuleList in the torch-derived ones. Both slice
        # to their own type. Anything else named `blocks` is not a stack and is left alone.
        if stack is None or isinstance(stack, (str, bytes)):
            continue
        try:
            n = len(stack)
        except TypeError:
            continue
        if n < 1:
            continue
        try:
            first = stack[0]
        except Exception:
            continue
        if first is None or isinstance(first, (int, float, str, bytes)):
            continue
        # Instrument in BOTH modes: the heartbeat is what keeps a slow block from being read as a
        # hang, and screen mode needs that as much as residency does.
        _instrument_class(type(first))
        if mode == "residency":
            continue
        if n < 2:
            continue                              # nothing to truncate; the shape still runs once
        try:
            setattr(obj, attr, stack[:1])
        except Exception as exc:                  # frozen dataclass, read-only property, ...
            _record(f"truncate {type(obj).__qualname__}.{attr}: {exc}")
            continue
        _state["truncated"].append([f"{type(obj).__module__}.{type(obj).__qualname__}", attr, n])


def _patch_module(mod) -> None:
    for name in dir(mod):
        cls = getattr(mod, name, None)
        if not isinstance(cls, type) or cls.__module__ != mod.__name__:
            continue
        init = cls.__dict__.get("__init__")
        if init is None or getattr(init, "_capacity_wrapped", False):
            continue

        def wrapped_init(self, *a, __init=init, **kw):
            __init(self, *a, **kw)
            try:
                _visit(self)
            except Exception as exc:
                _record(f"visit {type(self).__qualname__}: {type(exc).__name__}: {exc}")

        wrapped_init._capacity_wrapped = True
        try:
            cls.__init__ = wrapped_init
        except (AttributeError, TypeError):
            pass


class _Finder:
    """A meta-path finder that patches tt_bio modules the moment they finish executing.

    The model modules are imported lazily inside the CLI, well after this hook installs, so
    patching what is already in sys.modules is not enough.
    """

    def find_module(self, fullname, path=None):   # legacy API, harmless to omit
        return None

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("tt_bio"):
            return None
        if fullname.startswith(_SKIP_PREFIXES):
            return None
        # Ask everyone AFTER us to locate it, then wrap the loader they hand back.
        rest = [f for f in sys.meta_path if f is not self]
        for finder in rest:
            try:
                spec = finder.find_spec(fullname, path, target)
            except (AttributeError, TypeError):
                continue
            if spec is None or spec.loader is None:
                continue
            loader = spec.loader
            inner = loader.exec_module

            def exec_module(module, __inner=inner):
                __inner(module)
                try:
                    _patch_module(module)
                except Exception as exc:
                    _record(f"patch {module.__name__}: {type(exc).__name__}: {exc}")

            try:
                loader.exec_module = exec_module
            except AttributeError:
                return spec
            return spec
        return None


def install() -> None:
    """Arm the hook if TT_BIO_CAPACITY_HOOK names a mode. Called from the generated
    sitecustomize, so it runs in every interpreter the gate starts."""
    mode = (os.environ.get("TT_BIO_CAPACITY_HOOK") or "").strip()
    if mode not in ("screen", "residency"):
        return
    if _state["mode"] is not None:
        return
    _state["mode"] = mode
    global _beat_path
    beat = os.environ.get("TT_BIO_CAPACITY_HOOK_BEAT")
    _beat_path = f"{beat}.{os.getpid()}" if beat else None
    sys.meta_path.insert(0, _Finder())
    for name, mod in list(sys.modules.items()):
        if name.startswith("tt_bio") and not name.startswith(_SKIP_PREFIXES):
            try:
                _patch_module(mod)
            except Exception:
                pass
    atexit.register(_flush)
