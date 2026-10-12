"""ESMC page cell and accuracy with one tt_bio module attribute (--alt module.ATTR, a bool switch) on vs off.

One process, one model load; arms alternate off/on for --rounds so drift lands on both. Per arm: the page workload
(prot512.seq x7, one forward of 7, warm median of --repeat) and the first --length residues of --pdb scored against the
fp32 CPU reference (perf/mgx_embed/accuracy.py) and against the off arm of the same process.

    python3 perf/spd_gap_esmf/esmc_ab.py --alt tt_bio.esmc._ESMC_QKV_SPLIT --model esmc-600m --pdb perf/bgx_size/targets/hHSA.pdb \
        --ref-cache ../refcache --out ab.jsonl [--fast]
"""
import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", required=True, help="module.ATTR flipped between arms (off first)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--ref-cache", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=2)
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
    model = E.load_esmc(a.model, fast=a.fast, trace=False)
    one = lambda: E.embed_sequences(model, {"q": aa})[0].per_residue
    batch = {f"s{i}": page for i in range(7)}
    embed7 = lambda: E.embed_sequences(model, batch, batch_size=8)
    dev = model.tt_device

    base = None
    for r in range(a.rounds):
        for on in (False, True):
            setattr(alt_mod, attr, on)
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
            row = {"model": a.model, "fast": a.fast, "round": r, "alt": a.alt, "on": on,
                   "batch7_ms": round(sorted(ts)[len(ts) // 2] * 1000, 2),
                   "batch7_all_ms": [round(t * 1000, 2) for t in ts],
                   "vs_fp32": acc.metrics(out, ref), "vs_off": acc.metrics(out, base)}
            print(json.dumps(row), flush=True)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
