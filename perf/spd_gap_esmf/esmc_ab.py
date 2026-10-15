"""ESMC page cell and accuracy with one tt_bio module attribute (--alt module.ATTR, a bool switch) on vs off,
or (--fidelity NAME) the model's math fidelity at its default vs NAME.

One process, one model load; arms alternate off/on for --rounds so drift lands on both. Per arm: the page workload
(prot512.seq x7, one forward of 7, warm median of --repeat) and the first --length residues of --pdb scored against the
fp32 CPU reference (perf/mgx_embed/accuracy.py) and against the off arm of the same process.

    python3 perf/spd_gap_esmf/esmc_ab.py --alt tt_bio.esmc._ESMC_QKV_SPLIT --model esmc-600m --pdb perf/bgx_size/targets/hHSA.pdb \
        --ref-cache ../refcache --out ab.jsonl [--fast]

--fidelity sets math_fidelity on the compute kernel config every ESMC op shares, in place, so one load serves both
arms. If the flip did not reach the ops, the on arm's vs_off error is exactly 0.
"""
import argparse
import functools
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mgx_embed"))
import accuracy as acc  # noqa: E402

from tt_bio import esmc as E  # noqa: E402


def _blocks(obj, depth=3):
    """The transformer Block list of a loaded ESMC / ESMC-6B model, wherever it hangs."""
    if isinstance(getattr(obj, "blocks", None), list):
        return obj.blocks
    for v in (vars(obj).values() if depth and hasattr(obj, "__dict__") else ()):
        r = _blocks(v, depth - 1) if hasattr(v, "__dict__") and not isinstance(v, type) else None
        if r:
            return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", default="tt_bio.esmc._ESMC_QKV_SPLIT", help="module.ATTR flipped between arms (off first)")
    ap.add_argument("--fidelity", default=None, help="on arm = this ttnn.MathFidelity (HiFi4/HiFi3/HiFi2/LoFi); --alt unused")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--ref-cache", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--trace", action="store_true", help="load with ESMC.trace on (the served default; reserves the region)")
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--on-value", type=int, default=None, help="value the on arm sets (default True), for an int switch")
    ap.add_argument("--block-attr", default=None,
                    help="dotted attribute on every transformer Block (e.g. attn.qk_l1): off arm = as loaded, on = True")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    mod_name, attr = a.alt.rsplit(".", 1)
    alt_mod = importlib.import_module(mod_name)

    aa = acc.pdb_sequence(a.pdb)[: a.length]
    key = Path(a.ref_cache, f"{a.model}_{a.length}_seq.npy")
    if key.is_file():
        ref = np.load(key)
    else:
        ref = acc.esmc_reference(a.model)(aa, "#" * len(aa))
        key.parent.mkdir(parents=True, exist_ok=True)
        np.save(key, ref)

    page = Path(a.seq_file).read_text().strip()
    model = E.load_esmc(a.model, fast=a.fast, trace=a.trace)
    one = lambda: E.embed_sequences(model, {"q": aa})[0].per_residue
    batch = {f"s{i}": page for i in range(1 if a.model == "esmc-6b" else 7)}   # the GAP cell's batch
    embed7 = lambda: E.embed_sequences(model, batch, batch_size=8)
    dev = model.tt_device
    ckc = model.compute_kernel_config
    fid_off = ckc.math_fidelity

    battr = []
    if a.block_attr:   # set per Block instance: a value Block.__init__ derives is never re-read from the module
        *path, name = a.block_attr.split(".")
        battr = [(functools.reduce(getattr, path, b), name) for b in _blocks(model)]
        assert battr, "no transformer blocks found"
    boff = [getattr(o, n) for o, n in battr]
    cur = getattr(alt_mod, attr) if not (a.fidelity or a.block_attr) else None
    off_value = False if cur is None else type(cur)(0)   # tri-state switches (None = per-arch default) read off as False
    base = None
    for r in range(a.rounds):
        for on in (False, True):
            if a.fidelity:
                ckc.math_fidelity = getattr(ttnn.MathFidelity, a.fidelity) if on else fid_off
            elif a.block_attr:
                for (o, name), v in zip(battr, boff):
                    setattr(o, name, True if on else v)
            else:
                setattr(alt_mod, attr, (a.on_value if a.on_value is not None else True) if on else off_value)
            # a captured trace replays the program of the arm it was captured under: drop it so this arm recaptures
            if hasattr(model, "_release_traces"):   # ESMC-6B (ESMCLanguageModel) runs eager
                model._release_traces()
                model._trace_seen.clear()
            for _ in range(2):
                embed7()
            ts = []
            for _ in range(a.repeat):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                embed7()
                ttnn.synchronize_device(dev)
                ts.append(time.perf_counter() - t0)
            out = one()
            if base is None:
                base = out
            row = {"model": a.model, "fast": a.fast, "round": r, "alt": f"fidelity:{a.fidelity}" if a.fidelity else f"block:{a.block_attr}" if a.block_attr else a.alt, "on": on,
                   "batch7_ms": round(sorted(ts)[len(ts) // 2] * 1000, 2),
                   "batch7_all_ms": [round(t * 1000, 2) for t in ts],
                   "vs_fp32": acc.metrics(out, ref), "vs_off": acc.metrics(out, base),
                   "stats": {n: list(v) for n, v in vars(alt_mod).items() if n.endswith("_STATS") and isinstance(v, list)}}
            print(json.dumps(row), flush=True)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
