#!/usr/bin/env python3
"""Where BindCraft 2's device bytes go at the high-water mark, buffer by buffer.

    attrib.py --every 2 --step-mb 24 -- <perf/bgx_size/rung.py args, with --footprint>

`b2p-whtape` fitted the peak to `0.71 GB + 41.2 KB per token pair`. A coefficient is not
something a lever can be aimed at, so this run takes the same rung `perf/bgx_size/rung.py`
takes and, at every new high-water mark of the backward walk, lists every live DRAM buffer
the process holds with what it is:

  weight         a leaf registered as a parameter, or a raw ttnn handle with no N in it
  ckpt_pin       a checkpointed block's input, pinned from the forward to its recompute
  recompute_tape a value built inside a checkpoint's recompute (one block's working set)
  outer_tape     a taped value built outside any recompute (extra-MSA, template, embedding)
  grad:<role>    the gradient accumulator of a value of that role
  raw            a device handle no taped Tensor owns (constants, masks, kernel scratch)

and, for each buffer, the exponent of N in its shape (dims equal to the Evoformer axis, a
dim equal to N*N counted twice), the taped verb and model site that made it, and its bytes at
padded shape and stored dtype. The allocator's own `used` minus the sum of listed buffers is
reported as `unlisted`: buffers C++ holds that Python cannot name, plus page rounding.

Only DRAM buffers are counted. The point sampled is the walk's frontier, after a node's
closure has run and its own value has been retired, so it is the RESIDENT state between ops.
The transient inside one op (a kernel's output before its input is freed) is not visible
from Python; `transient_gap` reports the difference between the frontier maximum and the
allocator's request at a refusal when there is one.

INSTRUMENT DISTORTION, said here because it is the reason the numbers mean what they say:
`get_memory_view` drains the pipeline, so this leg reports no timing; and the gc walk runs
only when `used` rises `--step-mb` above the last walked sample, so the walked peak is within
that step of the true frontier peak, which is also recorded exactly (`frontier_peak_bytes`).
Nothing on disk is changed: the patches are module-level rebinds in this process.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import pathlib
import runpy
import sys
import traceback
import weakref

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

DT = {"FLOAT32": 4, "UINT32": 4, "INT32": 4, "BFLOAT16": 2, "UINT16": 2, "BFLOAT8_B": 1.0625,
      "BFLOAT4_B": 0.5625, "UINT8": 1}


class Census:
    def __init__(self, every: int, step: int, out: pathlib.Path):
        self.every, self.step, self.out = every, step, out
        self.calls = 0
        self.used_max = 0
        self.walked = 0
        self.best = None            # the walk at the highest `used` seen
        self.depth = 0              # recompute nesting
        self.block = 0              # recomputes entered, a proxy for the block index
        self.phase = "fwd"
        self.in_recompute: weakref.WeakSet = weakref.WeakSet()
        self.origin: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self.axis = None
        self.refusal = None
        self.history = []           # (calls, used) at every walk, to see the shape of the climb

    # -- patches -------------------------------------------------------------------------
    def install(self):
        from tt_bio import autograd, taped_ttnn
        census = self

        real_tape = autograd._tape

        def _tape(out_value, parents, make_fn, reads=None):
            out = real_tape(out_value, parents, make_fn, reads)
            try:
                f = sys._getframe(1)
                verb = f.f_code.co_name
                site = None
                while f is not None:
                    fn = f.f_code.co_filename
                    if not fn.endswith(("autograd.py", "taped_ttnn.py", "attrib.py")):
                        site = f"{os.path.basename(fn)}:{f.f_code.co_name}"
                        break
                    f = f.f_back
                census.origin[out] = (verb, site, census.phase, "pinall" if reads is None
                                      else "reads")
                if census.depth:
                    census.in_recompute.add(out)
            except Exception:                                              # noqa: BLE001
                pass
            return out
        # taped_ttnn imports `_tape` by name, so its verbs need their own rebind.
        autograd._tape = taped_ttnn._tape = _tape

        real_scope = taped_ttnn.recompute_scope

        class _Scope:
            def __enter__(self_):
                census.depth += 1
                census.block += 1
                self_.inner = real_scope()
                return self_.inner.__enter__()

            def __exit__(self_, *exc):
                census.depth -= 1
                return self_.inner.__exit__(*exc)
        taped_ttnn.recompute_scope = lambda: _Scope()

        real_backward = autograd._backward

        def _backward(roots, seeds):
            prev, census.phase = census.phase, "bwd"
            try:
                return real_backward(roots, seeds)
            finally:
                census.phase = prev
        autograd._backward = _backward

        real_retire = autograd._retire

        def _retire(t):
            r = real_retire(t)
            census.calls += 1
            if census.calls % census.every == 0:
                try:
                    census.sample()
                except Exception:                                          # noqa: BLE001
                    pass
            return r
        autograd._retire = _retire

    # -- sampling ------------------------------------------------------------------------
    def memview(self):
        import ttnn
        from tt_bio import tenstorrent
        if tenstorrent._device is None:
            return None
        mv = ttnn.get_memory_view(tenstorrent._device, ttnn.BufferType.DRAM)
        banks = int(mv.num_banks)
        total = int(mv.total_bytes_per_bank) * banks
        free = int(mv.total_bytes_free_per_bank) * banks
        lcf = mv.largest_contiguous_bytes_free_per_bank
        if isinstance(lcf, (list, tuple)):
            lcf = min(lcf)
        return total, total - free, int(lcf) * banks, banks

    def sample(self):
        m = self.memview()
        if m is None:
            return
        total, used, lcf, banks = m
        if used > self.used_max:
            self.used_max = used
        if used >= self.walked + self.step:
            self.walked = used
            rows = self.walk()
            self.history.append((self.calls, used))
            if self.best is None or used > self.best["used"]:
                self.best = {"used": used, "total": total, "largest_free_bytes": lcf,
                             "banks": banks, "calls": self.calls, "block": self.block,
                             "depth": self.depth, "rows": rows}
                self.flush()

    def _axis(self):
        if self.axis is None:
            try:
                import rung
                self.axis = rung.AXIS.get("evoformer_axis")
            except Exception:                                              # noqa: BLE001
                pass
        return self.axis

    def exponent(self, shape):
        n = self._axis()
        if not n:
            return None
        return sum(1 for d in shape if d == n) + 2 * sum(1 for d in shape if d == n * n) \
            + 3 * sum(1 for d in shape if d == n ** 3)

    def walk(self):
        import ttnn
        from tt_bio import autograd
        params = {id(t) for t in autograd._PARAMS.values()}
        pins = {id(t) for t in autograd._CKPT_PINS}
        seen, rows = set(), []

        def add(v, role, owner=None):
            try:
                if not isinstance(v, ttnn.Tensor) or v.storage_type() != ttnn.StorageType.DEVICE:
                    return
                if not v.is_allocated():
                    return
                if v.memory_config().buffer_type != ttnn.BufferType.DRAM:
                    return
                addr = v.buffer_address()
                if addr in seen:
                    return
                seen.add(addr)
                shp = tuple(int(d) for d in v.padded_shape)
                n = 1
                for d in shp:
                    n *= d
                dt = str(v.dtype).split(".")[-1]
                b = int(n * DT.get(dt, 2))
                o = self.origin.get(owner) if owner is not None else None
                rows.append({"bytes": b, "shape": list(shp), "logical": list(v.shape),
                             "dtype": dt, "role": role, "exp": self.exponent(shp),
                             "verb": o[0] if o else None, "site": o[1] if o else None,
                             "made_in": o[2] if o else None, "tape_rule": o[3] if o else None})
            except Exception:                                              # noqa: BLE001
                return

        objs = gc.get_objects()
        tensors = [o for o in objs if isinstance(o, autograd.Tensor)]
        for t in tensors:
            try:
                val = object.__getattribute__(t, "_value")
            except Exception:                                              # noqa: BLE001
                continue
            if id(t) in params:
                role = "weight"
            elif id(t) in pins:
                role = "ckpt_pin"
            elif t in self.in_recompute:
                role = "recompute_tape"
            elif t.node is not None:
                role = "outer_tape"
            else:
                role = "leaf_input"
            add(val, role, t)
            g = object.__getattribute__(t, "_grad")
            if g is not None:
                add(g, "grad:" + role, t)
            for part in (object.__getattribute__(t, "_parts") or []):
                add(part[2], "grad_part:" + role, t)
            box = object.__getattribute__(t, "box")
            if box:
                add(box[0], role + ":box", t)
        # A nanobind ttnn.Tensor is not gc-tracked, so it never appears in get_objects():
        # reach raw handles (weights, masks, constants) through whatever holds them.
        for o in objs:
            if isinstance(o, ttnn.Tensor):
                add(o, "raw")
                continue
            if isinstance(o, (dict, list, tuple, set)) or hasattr(o, "__dict__"):
                try:
                    for r in gc.get_referents(o):
                        if type(r) is ttnn.Tensor:
                            add(r, "raw")
                except Exception:                                          # noqa: BLE001
                    pass
        del objs, tensors
        return rows

    # -- report --------------------------------------------------------------------------
    def summary(self):
        if not self.best:
            return {"walks": 0}
        rows = self.best["rows"]
        listed = sum(r["bytes"] for r in rows)
        by_role, by_exp, by_key = collections.Counter(), collections.Counter(), {}
        for r in rows:
            role = r["role"]
            if role == "raw" and r["exp"] == 0:
                role = "raw_no_N"
            by_role[role] += r["bytes"]
            by_exp[str(r["exp"])] += r["bytes"]
            k = (role, r["exp"], r["verb"], r["site"], tuple(r["shape"]), r["dtype"])
            e = by_key.setdefault(k, [0, 0])
            e[0] += 1
            e[1] += r["bytes"]
        top = sorted(by_key.items(), key=lambda kv: -kv[1][1])
        b = self.best
        return {
            "axis": self._axis(), "frontier_peak_bytes": self.used_max,
            "walked_peak_used_bytes": b["used"], "device_total_bytes": b["total"],
            "largest_free_bytes_at_walk": b["largest_free_bytes"], "banks": b["banks"],
            "walk_at_retire_call": b["calls"], "recomputes_entered": b["block"],
            "recompute_depth": b["depth"], "listed_bytes": listed,
            "unlisted_bytes": b["used"] - listed, "n_buffers": len(rows),
            "by_role": dict(by_role.most_common()), "by_exponent": dict(by_exp),
            "groups": [{"role": k[0], "exp": k[1], "verb": k[2], "site": k[3],
                        "shape": list(k[4]), "dtype": k[5], "count": v[0], "bytes": v[1]}
                       for k, v in top[:200]],
            "history": self.history[-400:], "refusal": self.refusal,
        }

    def flush(self):
        try:
            (self.out / "attrib.json").write_text(json.dumps(self.summary(), indent=1))
        except Exception:                                                  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=2, help="sample at every Nth retired node")
    ap.add_argument("--step-mb", type=float, default=24.0,
                    help="walk the heap when DRAM use rises this far above the last walk")
    ap.add_argument("rung", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rargs = [x for x in a.rung if x != "--"]
    if "--footprint" not in rargs:
        rargs.append("--footprint")
    out = pathlib.Path(rargs[rargs.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "perf" / "bgx_size"))
    c = Census(a.every, int(a.step_mb * 2**20), out)
    c.install()
    sys.argv = [str(ROOT / "perf/bgx_size/rung.py")] + rargs
    rc = 0
    try:
        import rung
        rung.main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 0
    except BaseException as exc:                                           # noqa: BLE001
        c.refusal = "".join(traceback.format_exception_only(type(exc), exc))[-4000:]
        rc = 3
    c.flush()
    print(f"[attrib] frontier peak {c.used_max / 1e9:.3f} GB, walked "
          f"{(c.best or {}).get('used', 0) / 1e9:.3f} GB, rc={rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
