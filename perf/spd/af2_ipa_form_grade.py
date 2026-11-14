#!/usr/bin/env python3
"""Does 54ae49c54's IPA point-distance form move the AF2-IG structure? Card-free.

Runs the tap gate's host arm (`tap_gate.run_arm`, no card) on the gate's own complex fixture
twice per trunk dtype: once with the shipped |q|^2 + |k|^2 - 2 q.k form and once with the direct
[head, query, key, point, xyz] form it replaced. Prints the CA RMSD of every pass against the
JAX reference taps and between the two forms, in Angstrom.

    PYTHONPATH=. python3 perf/spd/af2_ipa_form_grade.py --params PATH [--dtypes bfloat16 float32]
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "af2_port"))

from tt_bio import af2_reference  # noqa: E402

NEW = """        centre = torch.cat([q_point, k_point], dim=1).mean(dim=(1, 2), keepdim=True)
        qf, kf = (q_point - centre).flatten(2), (k_point - centre).flatten(2)
        dist2 = (qf.square().sum(-1)[:, :, None] + kf.square().sum(-1)[:, None]
                 - 2 * qf @ kf.transpose(1, 2))
        logits = torch.einsum("hqc,hkc->hqk", scalar_weights * q_scalar, k_scalar)
        logits = logits - 0.5 * point_weights[:, None, None] * dist2
"""
OLD = """        dist2 = (q_point[:, :, None] - k_point[:, None]).square().sum(-1)
        logits = torch.einsum("hqc,hkc->hqk", scalar_weights * q_scalar, k_scalar)
        logits = logits - 0.5 * (point_weights[:, None, None, None] * dist2).sum(-1)
"""


def _direct_forward():
    src = inspect.getsource(af2_reference.InvariantPointAttention.forward)
    assert NEW in src, "the shipped IPA form changed; update NEW"
    ns = dict(vars(af2_reference))
    exec(textwrap.dedent(src.replace(NEW, OLD)), ns)
    return ns["forward"]


def _ca(pos) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64).reshape(-1, 37, 3)[:, 1]


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import tap_gate
    from tt_bio.af2_weights import load_af2_state_dict

    feats, prev = tap_gate.load_inputs(tap_gate.ARTIFACTS / "ref_inputs.npz")
    with np.load(tap_gate.ARTIFACTS / "ref_taps.npz") as z:
        jax = {i: _ca(z[f"structure_module#{i}/final_atom_positions/full"])
               for i in range(4) if f"structure_module#{i}/final_atom_positions/full" in z.files}
    state = load_af2_state_dict(args.params)
    shipped, direct = af2_reference.InvariantPointAttention.forward, _direct_forward()

    result = {}
    for dtype in args.dtypes:
        cas = {}
        for form, fwd in (("matmul", shipped), ("direct", direct)):
            af2_reference.InvariantPointAttention.forward = fwd
            outs = []
            for out in af2_reference.run_recycles(
                    af2_reference.load_af2_model(state, template=True, trunk_dtype=getattr(torch, dtype)),
                    feats, prev, num_recycles=tap_gate.DEFAULT_RECYCLES):
                outs.append(_ca(out["structure"]["final_atom_positions"].detach().float().numpy()))
            cas[form] = outs
        af2_reference.InvariantPointAttention.forward = shipped
        rows = []
        for i in range(len(cas["matmul"])):
            row = {"pass": i, "matmul_vs_direct": _rmsd(cas["matmul"][i], cas["direct"][i])}
            if i in jax:
                row["matmul_vs_jax"] = _rmsd(cas["matmul"][i], jax[i])
                row["direct_vs_jax"] = _rmsd(cas["direct"][i], jax[i])
            rows.append(row)
            print(dtype, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}),
                  flush=True)
        result[dtype] = rows
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
