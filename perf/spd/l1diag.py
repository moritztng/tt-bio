"""Name the L1 tensors that are live when a program's circular buffers clash with them.

A clash message gives one address and nothing else. With TT_BIO_L1DIAG=1, bench.py imports this before the model
and every ttnn op that raises a clash prints, on stderr under `[l1diag]`, each L1-resident ttnn.Tensor reachable from
the raising frames (locals and one level of `self` attributes) and from the garbage collector: where it is held,
shape, dtype, layout and buffer address. The tensor whose address the message names is the one to chase.
Diagnosis only; prints at most TT_BIO_L1DIAG_MAX (default 6) clashes per process.
"""
import gc
import os
import sys

import ttnn
from ttnn import decorators as _dec

_LEFT = [int(os.environ.get("TT_BIO_L1DIAG_MAX", "6"))]


def _l1(t):
    try:
        return t.is_allocated() and t.memory_config().buffer_type == ttnn.BufferType.L1
    except Exception:                                                          # noqa: BLE001
        return False


def _desc(t):
    try:
        addr = t.buffer_address()
    except Exception:                                                          # noqa: BLE001
        addr = "?"
    mc = t.memory_config()
    return f"addr={addr} shape={tuple(t.padded_shape)} {t.dtype} {t.layout} {mc.memory_layout}"


def dump(exc):
    if _LEFT[0] <= 0 or "clash" not in str(exc):
        return
    _LEFT[0] -= 1
    seen = {}
    tb = exc.__traceback__
    frames = []
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    f = frames[0].f_back if frames else None
    while f is not None:  # the callers above the op too
        frames.insert(0, f)
        f = f.f_back
    for fr in frames:
        where = f"{os.path.basename(fr.f_code.co_filename)}:{fr.f_lineno}:{fr.f_code.co_name}"
        for name, v in list(fr.f_locals.items()):
            cands = [(name, v)]
            if name == "self" and hasattr(v, "__dict__"):
                cands += [(f"self.{k}", w) for k, w in vars(v).items()]
            for n, w in cands:
                if isinstance(w, ttnn.Tensor) and _l1(w) and id(w) not in seen:
                    seen[id(w)] = (w, f"{where} {n}")
    try:
        for o in gc.get_objects():
            if isinstance(o, ttnn.Tensor) and _l1(o) and id(o) not in seen:
                seen[id(o)] = (o, "gc (no frame holds it)")
    except Exception:                                                          # noqa: BLE001
        pass
    msg = str(exc).split("backtrace")[0].strip().replace("\n", " ")
    print(f"[l1diag] CLASH {msg[-260:]}", file=sys.stderr, flush=True)
    for t, where in seen.values():
        print(f"[l1diag]   {_desc(t)}  held by {where}", file=sys.stderr, flush=True)
    for fr in frames[-12:]:
        print(f"[l1diag]   frame {os.path.basename(fr.f_code.co_filename)}:{fr.f_lineno} {fr.f_code.co_name}",
              file=sys.stderr, flush=True)


def _wrap(cls):
    orig = cls.__call__

    def __call__(self, *a, **kw):
        try:
            return orig(self, *a, **kw)
        except RuntimeError as e:
            dump(e)
            raise
    cls.__call__ = __call__


for _c in (_dec.FastOperation, _dec.Operation):
    _wrap(_c)
