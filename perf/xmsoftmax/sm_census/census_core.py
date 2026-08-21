"""Per-call-site row-sum accounting for `ttnn.softmax`, shared by the startup hook.

`ttnn.softmax` normalises against a denominator its own numerators do not match, so every row
it returns sums to less than 1 (0.9769 mean, 0.9613 min on [1,16,512,512] fp32, identical on
ttnn 0.67.4 / 0.68.0 / 0.75.0). That deficit is a multiplicative error on every weight in a
row, so it does not cancel in `probs @ v`. What it costs a given model depends on how peaked
that model's real logits are at each of its call sites, and a site whose rows already sum to
1.000 has nothing to win from a fix that costs 4.22x.

This records, per caller `file:line`, what the kernel actually returned during a real fold.
Row sums are reduced on device in fp32 and only the [..., 1] result comes back to host: a
[1,16,512,512] fp32 output is 16 MB and a fold makes thousands of calls.
"""
import json
import os
import sys
import traceback

_SELF = os.path.abspath(__file__)


class Census:
    def __init__(self, root: str, max_measured: int = 24):
        self.root = root
        self.max_measured = max_measured
        self.sites: dict[str, dict] = {}
        self.total = 0
        self.out_dir: str | None = None
        self.capture = None  # set by the startup hook when TT_BIO_SM_CAPTURE_DIR is set

    def site_of(self) -> str:
        for fr in reversed(traceback.extract_stack()[:-2]):
            fn = os.path.abspath(fr.filename)
            if fn == _SELF:
                continue
            rel = os.path.relpath(fn, self.root) if fn.startswith(self.root) else fn
            return f"{rel}:{fr.lineno}"
        return "<unknown>"

    def entry(self, site, shape, dtype, op) -> dict:
        key = f"{site}|{op}|{list(shape)}|{dtype}"
        e = self.sites.get(key)
        if e is None:
            e = self.sites[key] = {
                "site": site, "op": op, "shape": list(shape), "dtype": dtype,
                "n_calls": 0, "n_measured": 0, "rowsum_sum": 0.0,
                "rowsum_min": None, "rowsum_max": None, "rowsum_p01_min": None,
                "errors": [],
            }
        return e

    def measure(self, e, out) -> None:
        import ttnn
        import torch
        xf = t = None
        try:
            xf = out if out.dtype == ttnn.float32 else ttnn.typecast(out, ttnn.float32)
            t = ttnn.sum(xf, dim=-1, keepdim=True)
            rs = ttnn.to_torch(t).to(torch.float64).flatten()
            e["n_measured"] += 1
            e["rowsum_sum"] += float(rs.mean())
            lo, hi = float(rs.min()), float(rs.max())
            k = max(1, int(0.01 * rs.numel()))
            p01 = float(rs.kthvalue(k).values)
            e["rowsum_min"] = lo if e["rowsum_min"] is None else min(e["rowsum_min"], lo)
            e["rowsum_max"] = hi if e["rowsum_max"] is None else max(e["rowsum_max"], hi)
            e["rowsum_p01_min"] = (p01 if e["rowsum_p01_min"] is None
                                   else min(e["rowsum_p01_min"], p01))
        except Exception as exc:  # a census must never break the fold it measures
            if len(e["errors"]) < 3:
                e["errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            for tt in (t, xf):
                if tt is not None and tt is not out:
                    try:
                        ttnn.deallocate(tt)
                    except Exception:
                        pass

    def install(self) -> bool:
        """Wrap ttnn.softmax / softmax_in_place. False if ttnn is not ready yet."""
        import ttnn
        if not all(hasattr(ttnn, n) for n in
                   ("softmax", "softmax_in_place", "sum", "typecast", "float32")):
            return False
        for name in ("softmax", "softmax_in_place"):
            orig = getattr(ttnn, name)
            if getattr(orig, "_sm_census", False):
                continue
            setattr(ttnn, name, self._wrap(name, orig))
        return True

    def _wrap(self, name, orig):
        def wrapped(x, *a, **kw):
            out = orig(x, *a, **kw)
            if self.capture is not None:
                try:
                    self.capture.install_matmul_hooks()
                    self.capture.on_softmax(self.site_of(), x, out)
                except Exception:
                    pass
            try:
                dim = kw.get("dim", a[0] if a else -1)
                e = self.entry(self.site_of(), out.shape, str(out.dtype), name)
                e["n_calls"] += 1
                if dim in (-1, len(out.shape) - 1) and e["n_measured"] < self.max_measured:
                    self.measure(e, out)
                self.total += 1
                # tt_bio terminates its spawned workers rather than letting them exit, so
                # atexit does not fire in the process that did the work. Checkpoint instead.
                if self.out_dir and self.total % 128 == 0:
                    self.dump(self.out_dir)
            except Exception:
                pass
            return out
        wrapped.__name__ = name
        wrapped._sm_census = True
        return wrapped

    def dump(self, out_dir: str) -> None:
        rows = []
        for e in self.sites.values():
            e = dict(e)
            s, n = e.pop("rowsum_sum"), e["n_measured"]
            e["rowsum_mean"] = (s / n) if n else None
            e["deficit"] = (1.0 - e["rowsum_mean"]) if n else None
            rows.append(e)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"census_pid{os.getpid()}.json")
        payload = {"pid": os.getpid(), "argv": sys.argv, "sites": rows,
                   "tt_bio": _tt_bio_path()}
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)


def _tt_bio_path() -> str:
    m = sys.modules.get("tt_bio")
    return os.path.abspath(os.path.dirname(m.__file__)) if m and m.__file__ else "<none>"
