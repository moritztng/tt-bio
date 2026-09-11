#!/usr/bin/env python3
"""Runtime dtype census of tt-bio's Boltz-2 affinity path, on host, on CPU.

Moritz asked whether tt-bio's Boltz-2 carries fp32 islands of the kind BioIR deleted from
stock boltz. A grep over construction sites does not answer that (memory
`rf3-runs-bf16-on-gpu-kernel-counter-is-not-a-dtype`), so this runs the real module and
records what every tensor actually was.

The affinity module is the half of Boltz-2 that runs on the HOST in fp32 by default
(`BOLTZ2_AFFINITY_FP32_HOST`, default True -> `AffinityModule(use_tenstorrent=False)`), so
the census needs no Tenstorrent card: a TorchDispatchMode under a real forward sees every
aten op with its real operand and result dtypes.

Weights come from the shipped affinity checkpoint, so the module and its shapes are the
production ones. Op dtypes do not depend on weight values; wall times do not depend on them
either, and are reported for this host only.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import tt_bio.boltz2 as B

# Ops whose cost is a matmul, for a FLOP estimate that is counted rather than guessed.
_MM = {"mm", "bmm", "matmul", "addmm", "baddbmm", "linear", "einsum"}


class Census(TorchDispatchMode):
    """Record every aten call: name, operand dtypes, result dtype, result elements."""

    def __init__(self):
        self.rows = defaultdict(lambda: {"calls": 0, "out_elems": 0, "out_bytes": 0,
                                         "secs": 0.0, "flops": 0})
        self.depth = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func).split(".")[1] if "." in str(func) else str(func)
        ins = tuple(sorted({str(a.dtype).replace("torch.", "")
                            for a in args if isinstance(a, torch.Tensor)}))
        # Nested dispatch would double-count; only the outermost call is timed.
        top = self.depth == 0
        self.depth += 1
        t0 = time.perf_counter()
        try:
            out = func(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            self.depth -= 1
        if not top:
            return out
        outs = out if isinstance(out, (list, tuple)) else [out]
        od, elems, nbytes = "-", 0, 0
        for o in outs:
            if isinstance(o, torch.Tensor):
                od = str(o.dtype).replace("torch.", "")
                elems += o.numel()
                nbytes += o.numel() * o.element_size()
        key = (name, "|".join(ins), od)
        r = self.rows[key]
        r["calls"] += 1
        r["out_elems"] += elems
        r["out_bytes"] += nbytes
        r["secs"] += dt
        if name in _MM and len(args) >= 2 and all(isinstance(a, torch.Tensor) for a in args[:2]):
            a, b = args[0], args[1]
            if a.dim() >= 2 and b.dim() >= 2:
                r["flops"] += 2 * elems * a.shape[-1]
        return out


def make_feats(n_tokens, n_atoms, device, dtype):
    """The feature dict the affinity module reads, at production shapes."""
    lig = torch.zeros(1, n_tokens, dtype=torch.float32, device=device)
    lig[:, -8:] = 1.0                      # an 8-token ligand against the rest as receptor
    t2r = torch.zeros(1, n_tokens, n_atoms, dtype=dtype, device=device)
    idx = torch.arange(n_tokens, device=device) * (n_atoms // n_tokens)
    t2r[0, torch.arange(n_tokens, device=device), idx] = 1.0
    return {
        "token_pad_mask": torch.ones(1, n_tokens, dtype=dtype, device=device),
        "mol_type": torch.zeros(1, n_tokens, dtype=torch.long, device=device),
        "affinity_token_mask": lig,
        "token_to_rep_atom": t2r,
    }


def build(args_key, ckpt, token_s, token_z, dtype):
    hp = ckpt["hyper_parameters"]
    m = B.AffinityModule(token_s, token_z, use_tenstorrent=False, **hp[args_key])
    sd = {k.split(f"{args_key.replace('_args', '')}.", 1)[-1]: v
          for k, v in ckpt["state_dict"].items()}
    pref = "affinity_module1." if args_key.endswith("1") else "affinity_module2."
    own = {k[len(pref):]: v for k, v in ckpt["state_dict"].items() if k.startswith(pref)}
    if own:
        missing, unexpected = m.load_state_dict(own, strict=False)
        loaded = len(own) - len(unexpected)
    else:
        loaded = 0
    return m.to(dtype=dtype).eval(), loaded, len(list(m.state_dict()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--atoms", type=int, default=7168)
    ap.add_argument("--ckpt", default="/home/moritz/.boltz/boltz2_aff.ckpt")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--module", default="affinity_model_args1")
    ap.add_argument("--multiplicity", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(0)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    token_s, token_z = hp["token_s"], hp["token_z"]

    report = {"tokens": a.tokens, "multiplicity": a.multiplicity, "atoms": a.atoms, "module": a.module,
              "token_s": token_s, "token_z": token_z,
              "pairformer_blocks": hp[a.module]["pairformer_args"]["num_blocks"],
              "head_blocks": hp[a.module]["transformer_args"]["num_blocks"],
              "torch": torch.__version__, "arms": {}}

    for dname in a.dtypes.split(","):
        # The bf16 arm runs the SAME fp32-parameter module under CPU autocast rather than a
        # bf16 cast of the weights: torch's CPU LayerNorm refuses bf16 parameters, and a real
        # bf16 port would keep the norms in fp32 anyway. So this arm is "bf16 matmuls, fp32
        # norms", which is the arm a device bf16 affinity path would actually be.
        autocast = dname != "float32"
        dtype = torch.float32
        mod, loaded, total = build(a.module, ck, token_s, token_z, dtype)
        feats = make_feats(a.tokens, a.atoms, "cpu", dtype)
        s_in = torch.randn(1, a.tokens, token_s, dtype=dtype)
        z = torch.randn(1, a.tokens, a.tokens, token_z, dtype=dtype) * 0.1
        x = torch.randn(1, a.atoms, 3, dtype=dtype)

        with torch.no_grad():
            t0 = time.perf_counter()
            with Census() as c:
                if autocast:
                    with torch.autocast("cpu", dtype=getattr(torch, dname)):
                        out = mod(s_in, z, x, feats, multiplicity=a.multiplicity)
                else:
                    out = mod(s_in, z, x, feats, multiplicity=a.multiplicity)
            wall = time.perf_counter() - t0

        by_dtype = defaultdict(lambda: {"calls": 0, "secs": 0.0, "out_bytes": 0, "flops": 0})
        for (name, ins, od), r in c.rows.items():
            for k in ("calls", "secs", "out_bytes", "flops"):
                by_dtype[od][k] += r[k]
        top = sorted(c.rows.items(), key=lambda kv: -kv[1]["secs"])[:25]
        report["arms"][dname] = {
            "wall_s": round(wall, 4),
            "weights_loaded": f"{loaded}/{total}",
            "aten_calls": sum(r["calls"] for r in c.rows.values()),
            "by_out_dtype": {k: {"calls": v["calls"], "secs": round(v["secs"], 4),
                                 "out_GB": round(v["out_bytes"] / 1e9, 4),
                                 "TFLOP": round(v["flops"] / 1e12, 4)}
                             for k, v in sorted(by_dtype.items(), key=lambda kv: -kv[1]["secs"])},
            "top_ops": [{"op": k[0], "in": k[1], "out": k[2], "calls": v["calls"],
                         "secs": round(v["secs"], 4), "out_GB": round(v["out_bytes"] / 1e9, 4)}
                        for k, v in top],
            "out_keys": sorted(out.keys()),
        }
        print(f"{dname:10s} wall {wall:8.3f} s  aten {sum(r['calls'] for r in c.rows.values()):7d} "
              f"weights {loaded}/{total}", flush=True)
        for k, v in report["arms"][dname]["by_out_dtype"].items():
            print(f"    out={k:10s} calls {v['calls']:7d}  {v['secs']:8.3f} s  "
                  f"{v['out_GB']:8.3f} GB  {v['TFLOP']:7.3f} TFLOP", flush=True)

    Path(a.out).write_text(json.dumps(report, indent=1))
    print("WROTE " + a.out)


if __name__ == "__main__":
    main()
