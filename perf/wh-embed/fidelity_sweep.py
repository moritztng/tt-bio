"""Matmul fidelity vs speed and accuracy for one embed model, in one process.

The embed models run every linear at HiFi4 with fp32 accumulation (TorchWrapper's default). At HiFi4 the
fidelity passes, not operand bytes, set a compute-bound matmul's time, so this sweeps the fidelity itself.
Each variant swaps the shared compute-kernel config on every module of the loaded model, then
  * times the page workload: prot512.seq x7, one forward of 7 at the 576 bucket, warm median of --repeat
  * scores the first --length residues of --pdb against the model's fp32 CPU reference (perf/mgx_embed/accuracy.py)
    and against the HiFi4 output of the same process.

    python3 perf/wh-embed/fidelity_sweep.py --model esmc-300m --pdb perf/bgx_size/targets/hHSA.pdb \
        --ref-cache ../refcache --out sweep.jsonl [--fast]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mgx_embed"))
import accuracy as acc  # noqa: E402  (puts the repo root, scripts/ and tests/ on sys.path)

VARIANTS = [("HiFi4", True), ("HiFi3", True), ("HiFi2", True), ("LoFi", True), ("HiFi2", False), ("HiFi4", True)]


def swap_config(root, new):
    """Point every compute_kernel_config reachable from root at ``new``."""
    seen = set()

    def walk(o):
        if id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)
        elif type(o).__module__.startswith("tt_bio") and hasattr(o, "__dict__"):
            if "compute_kernel_config" in vars(o):
                o.compute_kernel_config = new
            for x in list(vars(o).values()):
                walk(x)

    walk(root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--ref-cache", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shipped-only", action="store_true", help="only the config the model was built with, no swap")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    saprot = a.model.startswith("saprot")

    aa = acc.pdb_sequence(a.pdb)[: a.length]
    key = Path(a.ref_cache, f"{a.model}_{a.length}_seq.npy")
    if key.is_file():
        ref = np.load(key)
    else:
        ref = (acc.saprot_reference if saprot else acc.esmc_reference)(a.model)(aa, "#" * len(aa))
        key.parent.mkdir(parents=True, exist_ok=True)
        np.save(key, ref)

    page = Path(a.seq_file).read_text().strip()
    if saprot:
        from tt_bio import saprot as mod
        model = mod.load_saprot(a.model, fast=a.fast)
        one = lambda: mod.embed_sequences(model, {"q": (aa, "#" * len(aa))})[0].per_residue
        batch = {f"s{i}": (page, "#" * len(page)) for i in range(7)}
    else:
        from tt_bio import esmc as mod
        model = mod.load_esmc(a.model, fast=a.fast, trace=False)
        one = lambda: mod.embed_sequences(model, {"q": aa})[0].per_residue
        batch = {f"s{i}": page for i in range(7)}
    embed7 = lambda: mod.embed_sequences(model, batch, batch_size=8)
    dev = model.tt_device
    kcls = type(model.compute_kernel_config)

    base = None
    shipped = model.compute_kernel_config
    for fid, acc32 in ([(str(shipped.math_fidelity).split(".")[-1], shipped.fp32_dest_acc_en)]
                       if a.shipped_only else VARIANTS):
        if not a.shipped_only:
            swap_config(model, kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                    fp32_dest_acc_en=acc32, packer_l1_acc=True))
        for _ in range(2):
            embed7()
        ts = []
        for _ in range(a.repeat):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            embed7()
            ts.append(time.perf_counter() - t0)
        out = one()
        if base is None:
            base = out
        row = {"model": a.model, "fast": a.fast, "fidelity": fid, "fp32_acc": acc32,
               "batch7_ms": round(sorted(ts)[len(ts) // 2] * 1000, 2), "batch7_all_ms": [round(t * 1000, 2) for t in ts],
               "vs_fp32": acc.metrics(out, ref), "vs_hifi4": acc.metrics(out, base)}
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
