#!/usr/bin/env python3
"""pair_transition op by op, device side, against op_up.py's float64 intermediates.

Each op is called the way `Transition._swiglu_all.swiglu` calls it (same weights, config, dtype
and core grid), fed the bf16 upload of the float64 value of its input. Then the residual add the
block lands the update with, `ttnn.add_` on two bf16 operands, scored against the exact sum of the
same two bf16 operands, beside round-to-nearest-even of that exact sum.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", type=Path, required=True)
    ap.add_argument("--boundary", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.path.expanduser("~/of3-weights/of3-p2-155k.pt")))
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as TT
    from tt_bio.openfold3_msa_embedder import MSAModule
    from tt_bio.openfold3_weights import is_openbind

    I = torch.load(a.inter, map_location="cpu", weights_only=False)
    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    n = int(B["outputs"].shape[-2])
    tokm = torch.diagonal(B["inputs"]["kwargs"]["pair_mask"].reshape(n, n)) > 0
    rz = lambda x: x.reshape(n, n, -1)[tokm][:, tokm]
    rel = lambda x, r: float((rz(x) - rz(r)).norm() / rz(r).norm())
    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    dev = TT.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    msa = MSAModule(sd, ckc, transpose_bias=not is_openbind(sd))
    del sd
    up = lambda x: ttnn.from_torch(x.float().reshape(1, n, n, -1), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    dn = lambda t: torch.Tensor(ttnn.to_torch(t)).double().reshape(1, n, n, -1)
    b16 = lambda x: x.to(torch.bfloat16).double()
    L1 = ttnn.L1_MEMORY_CONFIG
    rep = {"host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "board": open(f"/sys/class/tenstorrent/tenstorrent!{os.environ.get('TT_VISIBLE_DEVICES', '0')}/tt_card_type").read().strip(),
           "unfused_silu": bool(TT._UNFUSED_SILU), "linear_dtype": str(TT._dtype()),
           "blocks": {}}
    for i, blk in enumerate(msa.blocks):
        tr = blk.pair_stack.transition_z
        V = I[i]
        lin = lambda x, w, act=None, mc=L1: ttnn.linear(
            x, w, activation=act, compute_kernel_config=tr.compute_kernel_config,
            memory_config=mc, dtype=TT._dtype(), core_grid=TT.CORE_GRID_MAIN)
        ln = ttnn.layer_norm(up(V["x"]), weight=tr.norm_weight, bias=tr.norm_bias, epsilon=1e-5,
                             compute_kernel_config=tr.compute_kernel_config, memory_config=L1)
        f1 = dn(lin(up(V["ln"]), tr.fc1_weight))
        f2 = dn(lin(up(V["ln"]), tr.fc2_weight))
        # Which checkpoint linear each fc is, from the values rather than the remap's names.
        a_is_fc1 = rel(f1, V["a"]) < rel(f1, V["b"])
        wa, wb = (tr.fc1_weight, tr.fc2_weight) if a_is_fc1 else (tr.fc2_weight, tr.fc1_weight)
        ops = {"layer_norm": (dn(ln), V["ln"]),
               "linear_a": (f1 if a_is_fc1 else f2, V["a"]),
               "silu": (dn(ttnn.silu(up(V["a"]))), V["silu_a"]),
               "linear_a+silu": (dn(lin(up(V["ln"]), wa, "silu")), V["silu_a"]),
               "linear_b": (f2 if a_is_fc1 else f1, V["b"]),
               "gate_mul": (dn(ttnn.multiply_(up(V["silu_a"]), up(V["b"]))), V["h"]),
               "linear_out": (dn(lin(up(V["h"]), tr.fc3_weight, mc=ttnn.DRAM_MEMORY_CONFIG)),
                              V["out"])}
        row = {k: {"ours": rel(v, r), "bf16_floor": rel(b16(r), r)} for k, (v, r) in ops.items()}
        # The residual add. Operands are what the block holds: bf16 state, bf16 update.
        xs, us = b16(V["x"]), b16(V["out"])
        exact = xs + us
        for nm, fn in (("add_", lambda p, q: ttnn.add_(p, q)), ("add", lambda p, q: ttnn.add(p, q))):
            got = dn(fn(up(xs), up(us)))
            d = rz(got) - rz(exact)
            bias = float((d * torch.sign(rz(exact))).mean() / rz(exact).abs().mean())
            row[f"residual_{nm}"] = {
                "ours_vs_exact_sum_of_bf16_operands": rel(got, exact),
                "rne_of_exact_sum": rel(b16(exact), exact),
                "signed_bias_toward_larger_magnitude": bias,
                "frac_elements_equal_rne": float((rz(got) == rz(b16(exact))).double().mean()),
                "ours_vs_f64_state_plus_update": rel(got, V["x"] + V["out"])}
        rep["blocks"][i] = {"linear_a_is": "fc1" if a_is_fc1 else "fc2", "ops": row}
        print(f"b{i} (linear_a = {'fc1' if a_is_fc1 else 'fc2'})", flush=True)
        for k, r in row.items():
            if "ours" in r:
                print(f"   {k:14s} ours {r['ours']:.4e}  floor {r['bf16_floor']:.4e}")
            else:
                print(f"   {k:14s} vs exact {r['ours_vs_exact_sum_of_bf16_operands']:.4e}  RNE "
                      f"{r['rne_of_exact_sum']:.4e}  bias {r['signed_bias_toward_larger_magnitude']:+.3e}"
                      f"  ==RNE {r['frac_elements_equal_rne']:.4f}")
    a.report.write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
