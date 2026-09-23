#!/usr/bin/env python3
"""Every `ttnn.concat` on the OF3 trunk's forward and backward, with what its output costs.

of3t-crop768 recorded 576 tokens refusing a 2,717,908,992 B DRAM buffer inside
`ttnn::concat -> ttnn::prim::tilize_with_val_padding`, with 6,671,522,304 B free device-wide,
and left the call site unlocated. This wraps `ttnn.concat` and reports, per call site, the
operand shapes, the concat axis, the output shape and the bytes a tile-padded output costs.
The wrapper is host-side bookkeeping: it issues no device work, so it cannot move the bytes
it reports.

Calls collapse by (site, axis, operand specs), so one site entered 144 times is one row. A
call whose predicted output clears --stack-above-mb carries its tt_bio caller frames, which
is what turns a byte count into a file and a line. A refusal is recorded with the operands
that caused it before it is re-raised.

    concat_census.py --tokens 256 --out perf/of3t_cropwall/out/concat_256.json
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

ITEMSIZE = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2,
            "UINT8": 1, "BFLOAT8_B": 1, "BFLOAT4_B": 1}


def _tile_bytes(shape, dtype):
    """What a TILE-layout buffer of this logical shape costs: last two dims up to 32."""
    d = list(shape)
    if not d:
        return 0, ()
    d[-1] = -(-d[-1] // 32) * 32
    if len(d) > 1:
        d[-2] = -(-d[-2] // 32) * 32
    n = 1
    for x in d:
        n *= x
    return n * ITEMSIZE.get(dtype, 2), tuple(d)


def _spec(t):
    try:
        return (tuple(int(x) for x in t.shape),
                str(t.dtype).rsplit(".", 1)[-1].upper(),
                str(t.layout).rsplit(".", 1)[-1].upper())
    except Exception:                                                      # noqa: BLE001
        return (("?",), "?", "?")


def _site(depth=14):
    """The tt_bio / perf frames under the wrapper, innermost first."""
    out = []
    for f in reversed(traceback.extract_stack()[:-2]):
        p = f.filename
        if "/tt_bio/" in p or "/perf/" in p:
            out.append("%s:%d:%s" % (p.split("/")[-1], f.lineno, f.name))
        if len(out) >= depth:
            break
    return out


class Census:
    def __init__(self, stack_above_b):
        self.rows: dict = {}
        self.stack_above_b = stack_above_b
        self.phase = "pre"
        self.order = 0
        self.failed = None
        self.bookkeeping_error = None

    def note(self, specs, ax, out_spec, pad_b, pad_shape, site):
        k = (self.phase, ax, tuple(specs))
        r = self.rows.get(k)
        if r is None:
            self.order += 1
            r = self.rows[k] = {
                "phase": self.phase, "axis": ax, "first_seen": self.order,
                "operands": [{"shape": list(s[0]), "dtype": s[1], "layout": s[2]}
                             for s in specs[:4]],
                "n_operands": len(specs),
                "out_shape": list(out_spec[0]), "out_dtype": out_spec[1],
                "tile_padded_shape": list(pad_shape), "tile_padded_b": pad_b,
                "calls": 0, "site": site if pad_b >= self.stack_above_b else None,
            }
        r["calls"] += 1
        return r


def install(census):
    import ttnn
    real = ttnn.concat

    def wrapped(*a, **k):
        tensors = a[0] if a else k.get("tensors") or k.get("input_tensors")
        dim = a[1] if len(a) > 1 else k.get("dim", 0)
        specs = tuple(_spec(t) for t in tensors)
        rank = len(specs[0][0])
        ax = int(dim) % rank if rank else 0
        shape = list(specs[0][0])
        try:
            shape[ax] = sum(int(s[0][ax]) for s in specs)
        except Exception:                                                  # noqa: BLE001
            pass
        pad_b, pad_shape = _tile_bytes(shape, specs[0][1])
        try:
            site = _site() if pad_b >= census.stack_above_b else None
            row = census.note(specs, ax, (tuple(shape), specs[0][1], "TILE"), pad_b,
                              pad_shape, site)
        except Exception:                                                  # noqa: BLE001
            census.bookkeeping_error = traceback.format_exc()[-1200:]
            row = {"calls": -1}
        try:
            return real(*a, **k)
        except Exception as e:                                             # noqa: BLE001
            if census.failed is None:
                census.failed = {
                    "phase": census.phase, "axis": ax,
                    "operands": [{"shape": list(s[0]), "dtype": s[1], "layout": s[2]}
                                 for s in specs],
                    "n_operands": len(specs),
                    "out_shape": shape, "tile_padded_b": pad_b,
                    "tile_padded_shape": list(pad_shape),
                    "site": _site(24),
                    "error_head": str(e).split("backtrace")[0][:900],
                    "row_calls_before_failure": row["calls"],
                }
            raise

    ttnn.concat = wrapped
    from tt_bio import taped_ttnn as TT
    TT.forget_shim_bindings("concat")
    return real


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--dead-values", choices=("on", "off"), default=None)
    ap.add_argument("--stack-above-mb", type=float, default=32.0)
    ap.add_argument("--skip-backward", action="store_true")
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
    census = Census(int(a.stack_above_mb * (1 << 20)))

    def dump():
        rows = sorted(census.rows.values(), key=lambda r: -r["tile_padded_b"])
        out["concat"] = {"distinct": len(rows),
                         "calls": sum(r["calls"] for r in rows),
                         "bytes_x_calls_total": sum(r["tile_padded_b"] * r["calls"]
                                                    for r in rows),
                         "rows": rows[:60],
                         "failed": census.failed,
                         "bookkeeping_error": census.bookkeeping_error}
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
            z = None
            t0 = time.perf_counter()
            with ag.tape():
                _s, z = trunk(*args_, **kwargs_)
            ttnn.synchronize_device(dev)
            out["forward"] = {"cycles": a.cycles, "ok": True,
                              "s": round(time.perf_counter() - t0, 2)}
            dump()

            if not a.skip_backward:
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
                except Exception as e:                                     # noqa: BLE001
                    out["backward"]["ok"] = False
                    out["backward"]["error_head"] = str(e).split("backtrace")[0][:900]
                    out["backward"]["error"] = traceback.format_exc()[-4000:]
                finally:
                    rs.__exit__(None, None, None)
                out["backward"]["s"] = round(time.perf_counter() - t0, 2)
                out["backward"]["params_with_grad"] = sum(
                    1 for t in params.values() if getattr(t, "grad", None) is not None)
                out["backward"]["params_declared"] = len(params)
            ag.release_pins()
        except Exception:                                                  # noqa: BLE001
            out["error"] = traceback.format_exc()[-4000:]
    out["env"]["aiclk_during"] = clk.summary()
    out["env"]["aiclk_line"] = clk.line(0)
    dump()

    c = out["concat"]
    print("tokens %d  concat sites %d  calls %d" % (a.tokens, c["distinct"], c["calls"]),
          flush=True)
    for r in c["rows"][:12]:
        print("  %-8s ax%-2d n_op %-3d out %-26s padded %14d B  x%d  %s"
              % (r["phase"], r["axis"], r["n_operands"], str(r["out_shape"]),
                 r["tile_padded_b"], r["calls"], (r["site"] or [""])[0]), flush=True)
    if c["failed"]:
        f = c["failed"]
        print("REFUSED in concat: %d B padded, out %s, %d operands, axis %d"
              % (f["tile_padded_b"], f["out_shape"], f["n_operands"], f["axis"]), flush=True)
        for s in f["site"][:8]:
            print("    at", s, flush=True)
    print(out["env"]["aiclk_line"], flush=True)
    if out.get("error"):
        print("ERROR:", out["error"][-1500:], flush=True)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
