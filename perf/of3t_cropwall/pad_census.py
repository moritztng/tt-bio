#!/usr/bin/env python3
"""Every device tensor the OF3 trunk's forward and backward produce, ranked by padding waste.

`concat_census.py` ranked concat sites by ABSOLUTE bytes, which is how the 576 wall was found.
The defect it found is a RATIO: a TILE-layout buffer pads its last two dims to 32, so an op
whose output has extent e < 32 in the second-to-last dim allocates 32/e times its payload. A
site at 32x on 67 MB is a better fix than one at 1.03x on 2 GB, and once the largest offender
is gone every remaining site's share of the peak grows. So this ranks by allocated / logical.

It wraps every `ttnn` and `ttnn.experimental` operation and, after each OUTERMOST call, reads
each returned device tensor's logical and padded shape. Host-side bookkeeping only: it issues
no device work, so it cannot move the bytes it reports. An op's internal intermediates are not
seen, only what it returns; that is what stays live and what the peak is made of.

    pad_census.py --tokens 256 --out perf/of3t_cropwall/out/pad_256.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                                    # noqa: E402
from perf.of3t_perf import step as S                                   # noqa: E402

# Bytes per element as stored. Block-float formats carry one shared exponent per 16 values.
ITEMSIZE = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1,
            "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625}
# Frames that are dispatch plumbing rather than a call site: the shim's generic wrappers, and
# (by prefix, below) its per-verb forward entries `_v_*`. A vjp closure defined inside one of
# those has its own name, so a backward site still resolves to taped_ttnn.py.
_PLUMBING = {"call", "wrapped", "_taped_verb"}


def _site():
    """The innermost tt_bio frame that is not dispatch plumbing, as file:line:function."""
    f = sys._getframe(2)
    while f is not None:
        p = f.f_code.co_filename
        n = f.f_code.co_name
        if "/tt_bio/" in p and n not in _PLUMBING \
                and not (p.endswith("taped_ttnn.py") and n.startswith("_v_")):
            return "%s:%d:%s" % (p.split("/")[-1], f.f_lineno, f.f_code.co_name)
        f = f.f_back
    return "?"


def _tensors(r):
    if isinstance(r, (list, tuple)):
        for x in r:
            yield from _tensors(x)
    elif r is not None:
        v = getattr(r, "value", None)          # an autograd.Tensor carries the ttnn one
        yield v if v is not None and hasattr(v, "padded_shape") else r


class Census:
    def __init__(self):
        self.rows: dict = {}
        self.phase = "pre"
        self.depth = 0
        self.bookkeeping_error = None

    def note(self, op, site, t, inputs):
        try:
            if not hasattr(t, "padded_shape") or "DEVICE" not in str(t.storage_type()) \
                    or not t.is_allocated() or t.buffer_address() in inputs:
                return          # host tensor, freed, or a view / in-place result of an input
            shape = tuple(int(x) for x in t.shape)
            padded = tuple(int(x) for x in t.padded_shape)
            dt = str(t.dtype).rsplit(".", 1)[-1].upper()
            mem = str(t.memory_config().buffer_type).rsplit(".", 1)[-1].upper()
        except Exception:                                                  # noqa: BLE001
            self.bookkeeping_error = traceback.format_exc()[-1200:]
            return
        k = (self.phase, op, site, shape, dt, mem)
        r = self.rows.get(k)
        if r is None:
            item = ITEMSIZE.get(dt, 2)
            n_l = n_p = 1
            for x in shape:
                n_l *= x
            for x in padded:
                n_p *= x
            r = self.rows[k] = {"phase": self.phase, "op": op, "site": site,
                                "shape": list(shape), "padded": list(padded), "dtype": dt,
                                "mem": mem, "logical_b": int(n_l * item),
                                "allocated_b": int(n_p * item), "calls": 0}
            r["ratio"] = round(r["allocated_b"] / max(r["logical_b"], 1), 4)
        r["calls"] += 1


def install(census):
    """Wrap every operation on `ttnn` and `ttnn.experimental`; return an undo list."""
    import ttnn
    from tt_bio import taped_ttnn as TT
    undo = []
    for ns, prefix in ((ttnn, ""), (ttnn.experimental, "experimental.")):
        for name in dir(ns):
            f = getattr(ns, name, None)
            if name.startswith("_") or type(f).__name__ not in ("FastOperation", "Operation"):
                continue

            def wrapped(*a, _f=f, _q=prefix + name, **k):
                census.depth += 1
                try:
                    r = _f(*a, **k)
                finally:
                    census.depth -= 1
                if census.depth == 0:
                    site = _site()
                    inputs = set()
                    for t in _tensors(list(a) + list(k.values())):
                        try:
                            inputs.add(t.buffer_address())
                        except Exception:                                  # noqa: BLE001
                            pass
                    for t in _tensors(r):
                        census.note(_q, site, t, inputs)
                return r
            setattr(ns, name, wrapped)
            undo.append((ns, name, f))
    # The shim cached closures over the shipped callables during capture; drop them so the
    # taped forward resolves the wrapped ones.
    TT._SHIM.__dict__.clear()
    object.__setattr__(TT._SHIM, "_real", ttnn)
    object.__setattr__(TT._SHIM, "_prefix", "")
    return undo


def summarise(rows, min_b):
    """Site-level totals, then the rankings the defect needs: by ratio, and by waste."""
    sites: dict = {}
    for r in rows:
        s = sites.setdefault((r["phase"], r["op"], r["site"]), {
            "phase": r["phase"], "op": r["op"], "site": r["site"], "calls": 0,
            "logical_b_x_calls": 0, "allocated_b_x_calls": 0, "largest_allocated_b": 0,
            "worst_shape": None, "worst_padded": None, "worst_ratio": 0.0, "dtype": r["dtype"]})
        s["calls"] += r["calls"]
        s["logical_b_x_calls"] += r["logical_b"] * r["calls"]
        s["allocated_b_x_calls"] += r["allocated_b"] * r["calls"]
        s["largest_allocated_b"] = max(s["largest_allocated_b"], r["allocated_b"])
        if r["allocated_b"] >= min_b and r["ratio"] > s["worst_ratio"]:
            s["worst_ratio"], s["worst_shape"], s["worst_padded"] = \
                r["ratio"], r["shape"], r["padded"]
    for s in sites.values():
        s["ratio"] = round(s["allocated_b_x_calls"] / max(s["logical_b_x_calls"], 1), 4)
        s["waste_b_x_calls"] = s["allocated_b_x_calls"] - s["logical_b_x_calls"]
    big = [s for s in sites.values() if s["largest_allocated_b"] >= min_b]
    tot_a = sum(s["allocated_b_x_calls"] for s in sites.values())
    tot_l = sum(s["logical_b_x_calls"] for s in sites.values())
    return {
        "sites": len(sites), "calls": sum(s["calls"] for s in sites.values()),
        "allocated_b_x_calls": tot_a, "logical_b_x_calls": tot_l,
        "overall_ratio": round(tot_a / max(tot_l, 1), 4),
        "sites_over_min_b": len(big), "min_b": min_b,
        "top_by_ratio": sorted(big, key=lambda s: (-s["ratio"], -s["waste_b_x_calls"]))[:40],
        "top_by_waste": sorted(sites.values(), key=lambda s: -s["waste_b_x_calls"])[:25],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--dead-values", choices=("on", "off"), default=None)
    ap.add_argument("--min-mb", type=float, default=1.0,
                    help="a site enters the ratio ranking only if one of its outputs is at "
                         "least this big; a [1,1,1,1] scalar is 1024x padded and irrelevant")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    census = Census()
    min_b = int(a.min_mb * (1 << 20))

    def dump():
        out["census"] = summarise(list(census.rows.values()), min_b)
        out["census"]["bookkeeping_error"] = census.bookkeeping_error
        a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    with during() as clk:
        try:
            import ttnn
            from tt_bio import autograd as ag
            if a.dead_values is not None:
                ag.DROP_DEAD_VALUES = a.dead_values == "on"
            out["env"]["drop_dead_values"] = bool(getattr(ag, "DROP_DEAD_VALUES", False))
            from tt_bio import taped_ttnn as TT
            from tt_bio.tenstorrent import get_device

            held, _meta = S.capture(a.tokens, out)
            trunk = held["trunk"][0]
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            params = S.declare_weights(trunk, out)
            install(census)
            gc.collect()

            snap_args, snap_kwargs = held["trunk_snap"]
            args_ = S._rehydrate(snap_args, dev)
            kwargs_ = {k: v for k, v in S._rehydrate(snap_kwargs, dev).items()
                       if k != "progress_fn"}
            trunk.num_cycles = a.cycles

            census.phase = "forward"
            t0 = time.perf_counter()
            with ag.tape():
                _s, z = trunk(*args_, **kwargs_)
            ttnn.synchronize_device(dev)
            out["forward"] = {"cycles": a.cycles, "ok": True,
                              "s": round(time.perf_counter() - t0, 2)}
            dump()

            import torch
            zr = z.value
            seed = ttnn.from_torch(torch.ones(tuple(int(d) for d in zr.shape)),
                                   layout=ttnn.TILE_LAYOUT, device=dev, dtype=zr.dtype)
            census.phase = "backward"
            out["backward"] = {}
            t0 = time.perf_counter()
            rs = TT.recompute_scope()
            rs.__enter__()
            try:
                ag.backward([z], [seed])
                ttnn.synchronize_device(dev)
                out["backward"]["ok"] = True
            except Exception as e:                                         # noqa: BLE001
                out["backward"]["ok"] = False
                out["backward"]["error"] = traceback.format_exc()[-4000:]
            finally:
                rs.__exit__(None, None, None)
            out["backward"]["s"] = round(time.perf_counter() - t0, 2)
            out["backward"]["params_with_grad"] = sum(
                1 for t in params.values() if getattr(t, "grad", None) is not None)
            ag.release_pins()
        except Exception:                                                  # noqa: BLE001
            out["error"] = traceback.format_exc()[-4000:]
    out["env"]["aiclk_during"] = clk.summary()
    out["env"]["aiclk_line"] = clk.line(0)
    dump()

    c = out["census"]
    print("tokens %d  sites %d  calls %d  overall allocated/logical %.4f"
          % (a.tokens, c["sites"], c["calls"], c["overall_ratio"]), flush=True)
    print("top by ratio (sites with an output >= %.0f MB):" % a.min_mb, flush=True)
    for s in c["top_by_ratio"][:20]:
        print("  %8.3fx  %-8s %-34s %-44s waste %14d B  x%d  %s -> %s"
              % (s["ratio"], s["phase"], s["op"], s["site"], s["waste_b_x_calls"],
                 s["calls"], s["worst_shape"], s["worst_padded"]), flush=True)
    print("top by waste:", flush=True)
    for s in c["top_by_waste"][:12]:
        print("  %8.3fx  %-8s %-34s %-44s waste %14d B  x%d"
              % (s["ratio"], s["phase"], s["op"], s["site"], s["waste_b_x_calls"],
                 s["calls"]), flush=True)
    print(out["env"]["aiclk_line"], flush=True)
    if out.get("error"):
        print("ERROR:", out["error"][-1500:], flush=True)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
