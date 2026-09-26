#!/usr/bin/env python3
"""`AdamW.step()` before and after this row, interleaved in ONE card visit. The row's A/B.

Both optimizers live in the same process, on the same parameter tensors, alternating arms, so
they share the card, the page cache and whatever the host's load is doing. That matters more
here than the usual A/A hygiene: the same host arithmetic reads 0.485 s at 17 GiB MemAvailable
and 1.44-1.69 s at 1 GiB (`perf/of3t_p10host/balloon.py`), so two arms taken minutes apart on
this box are not comparable and two taken in alternation are.

The old optimizer is loaded from `42a664fa6^` through `importlib`, not by editing the tree, so
nothing in the worktree moves while a card is open.

Census: OF3T's own, 3,152 tensors over 381.3 M elements, at near-square 2D shapes. NOT row
vectors -- `TILE_LAYOUT` pads a single row to 32, and a `(1, n)` census measures 32x the bytes
a real weight crosses (read 6.3 s / write 27.1 s against 0.33 s / 0.74 s).
"""
from __future__ import annotations

import argparse, importlib.util, json, subprocess, sys, time, types
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf/of3t_p10host"))
from optsplit import census                                          # noqa: E402

OLD_AT = "42a664fa6^"


def load_old():
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{OLD_AT}:tt_bio/train/optim.py"],
                         capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("optim_old")
    mod.__package__ = "tt_bio.train"
    mod.__file__ = str(REPO / "tt_bio/train/optim.py")
    exec(compile(src, "optim_old", "exec"), mod.__dict__)
    return mod


class Leaf:
    """What `AdamW` asks of a parameter: a device `value` and a `grad`. Identical for both."""
    __slots__ = ("value", "grad")

    def __init__(self, value, grad):
        self.value, self.grad = value, grad


def aiclk(card="0"):
    p = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3, help="old/new pairs, alternating")
    ap.add_argument("--card", default="0")
    ap.add_argument("--out", type=Path, default=REPO / "perf/of3t_p10host/out/optab.json")
    a = ap.parse_args()

    import ttnn
    from tt_bio.train import optim as new_mod
    from tt_bio.train.tensors import to_device
    old_mod = load_old()

    shapes = census()
    nel = sum(r * c for r, c in shapes)
    print(f"census {len(shapes)} tensors, {nel:,} elements", flush=True)

    rng = np.random.default_rng(0)
    dev = ttnn.open_device(device_id=0)
    clk, rows = [], []
    try:
        params = {}
        for i, (r, c) in enumerate(shapes):
            w = (rng.standard_normal((r, c)) * 0.02).astype(np.float32)
            g = (rng.standard_normal((r, c)) * 1e-3).astype(np.float32)
            params[f"p{i}"] = Leaf(to_device(w, dev), to_device(g, dev))
        ttnn.synchronize_device(dev)
        print("uploaded", flush=True)

        # Two optimizers over the SAME leaves. Each step moves the weights, so the arms are
        # not bit-comparable across rounds -- that is not what this measures. What it
        # measures is the cost of a step at this census, and both arms pay it on the same
        # tensors in the same process.
        opts = {"old": old_mod.AdamW(params, lr=3e-4),
                "new": new_mod.AdamW(params, lr=3e-4)}
        for rnd in range(a.rounds):
            for arm in ("old", "new"):
                clk.append(aiclk(a.card))
                t0 = time.perf_counter()
                opts[arm].step()
                s = time.perf_counter() - t0
                clk.append(aiclk(a.card))
                skipped = getattr(opts[arm], "last_writes_skipped", None)
                rows.append({"round": rnd, "arm": arm, "step_s": s, "writes_skipped": skipped})
                print(f"round {rnd} {arm:3s} step {s:7.3f}s  writes_skipped={skipped}",
                      flush=True)
    finally:
        ttnn.close_device(dev)

    import statistics as st
    med = {arm: st.median([r["step_s"] for r in rows if r["arm"] == arm])
           for arm in ("old", "new")}
    got = [c for c in clk if c]
    out = {"census": {"tensors": len(shapes), "elements": nel}, "rows": rows,
           "median_s": med, "delta_s": med["old"] - med["new"],
           "ratio": med["old"] / med["new"],
           "aiclk_during": {"min": min(got), "max": max(got),
                            "median": int(st.median(got)), "n": len(got)} if got else None,
           "mem_available_gib": round(
               [int(l.split()[1]) for l in open("/proc/meminfo")
                if l.startswith("MemAvailable")][0] / 1048576, 2),
           "loadavg": open("/proc/loadavg").read().split()[:3]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nold {med['old']:.3f}s -> new {med['new']:.3f}s  "
          f"delta {out['delta_s']:+.3f}s  {out['ratio']:.4f}x")
    print(f"AICLK during: {out['aiclk_during']}  MemAvailable "
          f"{out['mem_available_gib']} GiB  loadavg {out['loadavg'][0]}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
