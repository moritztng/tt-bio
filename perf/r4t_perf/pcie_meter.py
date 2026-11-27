"""Bytes and transfers over PCIe per trunk_step phase (D70: the Galaxy DP derate is per-rank host<->device traffic).

Counts every host->device upload and device->host read that goes through ttnn's Python entry points, keyed by the
phase the caller sets (`phase("fwd")` ...). Bytes are the on-card storage size (bfloat8_b 1088/1024 B per element,
bfloat4_b 576/1024), which is what crosses the link for a tiled transfer. Measurement only: it wraps the entry points,
the arithmetic is unchanged.
    from pcie_meter import install, phase, take
"""
import collections
import threading

import ttnn

_LOCK = threading.Lock()
_PHASE = ["other"]
_TAB = collections.defaultdict(lambda: [0, 0, 0, 0])   # phase -> [up_B, up_n, down_B, down_n]
_SIZE = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "UINT32": 4.0, "INT32": 4.0, "UINT16": 2.0, "UINT8": 1.0,
         "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024}
_ON = [False]


def _nbytes(t, dtype=None):
    try:
        vol = int(t.logical_volume()) if hasattr(t, "logical_volume") else int(t.volume())
    except Exception:
        vol = int(t.numel()) if hasattr(t, "numel") else 0
    d = dtype if dtype is not None else getattr(t, "dtype", None)
    return int(vol * _SIZE.get(str(d).split(".")[-1], 4.0))


def _on_device(t):
    try:
        return t.storage_type() == ttnn.StorageType.DEVICE
    except Exception:
        return False


def _count(i, b):
    with _LOCK:
        row = _TAB[_PHASE[0]]
        row[i] += b
        row[i + 1] += 1


def install():
    if _ON[0]:
        return
    _ON[0] = True
    f_torch, f_todev, f_copy, f_fromdev = ttnn.from_torch, ttnn.to_device, ttnn.copy_host_to_device_tensor, ttnn.from_device

    def from_torch(t, *a, **k):
        out = f_torch(t, *a, **k)
        if k.get("device") is not None and _on_device(out):
            _count(0, _nbytes(out))
        return out

    def to_device(t, *a, **k):
        _count(0, _nbytes(t))
        return f_todev(t, *a, **k)

    def copy_h2d(src, dst, *a, **k):
        _count(0, _nbytes(src, getattr(dst, "dtype", None)))
        return f_copy(src, dst, *a, **k)

    def from_device(t, *a, **k):
        _count(2, _nbytes(t))
        return f_fromdev(t, *a, **k)

    ttnn.from_torch, ttnn.to_device, ttnn.copy_host_to_device_tensor, ttnn.from_device = from_torch, to_device, copy_h2d, from_device
    try:   # Tensor.cpu() is the GIL-free read hostlane uses; ttnn.from_device does not go through it
        cpu = ttnn.Tensor.cpu

        def cpu_counted(self, *a, **k):
            if _on_device(self):
                _count(2, _nbytes(self))
            return cpu(self, *a, **k)
        ttnn.Tensor.cpu = cpu_counted
    except Exception:
        pass


def phase(name):
    _PHASE[0] = name


def take():
    """{phase: {up_MB, up_n, down_MB, down_n}} since the last take, and reset."""
    with _LOCK:
        out = {p: dict(up_MB=round(r[0] / 1e6, 2), up_n=r[1], down_MB=round(r[2] / 1e6, 2), down_n=r[3])
               for p, r in _TAB.items() if r[1] or r[3]}
        _TAB.clear()
    return out
