"""Bytes and transfers over PCIe per trunk_step phase (D70: the Galaxy DP derate is per-rank host<->device traffic).

Counts every host->device upload and device->host read that goes through ttnn's Python entry points, keyed by the
phase the caller sets (`phase("fwd")` ...). Bytes are the on-card storage size (bfloat8_b 1088/1024 B per element,
bfloat4_b 576/1024), which is what crosses the link for a tiled transfer. Measurement only: it wraps the entry points,
the arithmetic is unchanged.
    from pcie_meter import install, phase, take
"""
import collections
import os
import sys
import threading

import ttnn

_LOCK = threading.Lock()
_PHASE = ["other"]
_TAB = collections.defaultdict(lambda: [0, 0, 0, 0])   # phase -> [up_B, up_n, down_B, down_n]
_SIZE = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "UINT32": 4.0, "INT32": 4.0, "UINT16": 2.0, "UINT8": 1.0,
         "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024}
_ON = [False]
_SITES = [None]   # phase -> {(site, shape, dtype, dir): [B, n]} when install(sites=True)
_SKIP = ("pcie_meter.py", os.sep + "ttnn" + os.sep, "hostlane.py")


def _site():
    f = sys._getframe(2)
    while f is not None and any(k in f.f_code.co_filename for k in _SKIP):
        f = f.f_back
    if f is None:
        return "?"
    g = f.f_back
    up = f"<{os.path.basename(g.f_code.co_filename)}:{g.f_lineno}" if g is not None else ""
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}:{f.f_code.co_name}{up}"


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


def _count(i, b, t=None):
    key = None
    if _SITES[0] is not None:
        key = (_site(), str(tuple(t.shape)) if t is not None else "", str(getattr(t, "dtype", "")).split(".")[-1],
               "up" if i == 0 else "down")
    with _LOCK:
        row = _TAB[_PHASE[0]]
        row[i] += b
        row[i + 1] += 1
        if key is not None:
            c = _SITES[0].setdefault(_PHASE[0], {}).setdefault(key, [0, 0])
            c[0] += b
            c[1] += 1


def install(sites=False):
    if _ON[0]:
        return
    if sites:
        _SITES[0] = {}
    _ON[0] = True
    f_torch, f_todev, f_copy, f_fromdev = ttnn.from_torch, ttnn.to_device, ttnn.copy_host_to_device_tensor, ttnn.from_device

    def from_torch(t, *a, **k):
        out = f_torch(t, *a, **k)
        if k.get("device") is not None and _on_device(out):
            _count(0, _nbytes(out), out)
        return out

    def to_device(t, *a, **k):
        _count(0, _nbytes(t), t)
        return f_todev(t, *a, **k)

    def copy_h2d(src, dst, *a, **k):
        _count(0, _nbytes(src, getattr(dst, "dtype", None)), dst)
        return f_copy(src, dst, *a, **k)

    def from_device(t, *a, **k):
        _count(2, _nbytes(t), t)
        return f_fromdev(t, *a, **k)

    ttnn.from_torch, ttnn.to_device, ttnn.copy_host_to_device_tensor, ttnn.from_device = from_torch, to_device, copy_h2d, from_device
    try:   # Tensor.cpu() is the GIL-free read hostlane uses; ttnn.from_device does not go through it
        cpu = ttnn.Tensor.cpu

        def cpu_counted(self, *a, **k):
            if _on_device(self):
                _count(2, _nbytes(self), self)
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
        if _SITES[0]:
            out["sites"] = {p: [dict(site=k[0], shape=k[1], dtype=k[2], dir=k[3], MB=round(v[0] / 1e6, 2), n=v[1])
                                for k, v in sorted(d.items(), key=lambda kv: -kv[1][0])[:25]]
                            for p, d in _SITES[0].items()}
            _SITES[0] = {}
    return out
