#!/usr/bin/env python3
"""The confidence head's global layer norm, both arms, on REAL trunk output.

`gln_ab.py` scored this on `torch.randn`. Off-manifold inputs are exactly what made pass
3's triangle-attention lever look like a 5x regression when it was neutral, so the arm
that decides whether the row fold ships has to see the tensors a fold actually produces.
Three measurements, one process, one checkpoint load:

  1. op level. Both device arms against an fp64 `F.layer_norm(x, normalized_shape=x.shape)`
     on the SAME bf16 values, so the input quantisation is common to every arm and only the
     reduction's own error is left.
  2. head level. There is no torch golden for this head here, so one is built on device:
     the exact fp64 normalisation is injected in place of `global_layer_norm` and the real
     head run on it. That is what each arm's four logit tensors are then scored against.
  3. time, arms interleaved, arm A twice for the A/A floor.
"""
from __future__ import annotations

import argparse
import enum
import json
import sys
import time
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

EPS = 1e-5


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).pow(2).mean().sqrt() / b.double().std())


def exact_gln(x_bf16_as_f64: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x_bf16_as_f64, normalized_shape=tuple(x_bf16_as_f64.shape),
                        eps=EPS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=2)
    ap.add_argument("--num_steps", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3 import confidence_head as ch
    from tt_bio.rf3.featurize import featurize
    from tt_bio.rf3.host import HostInputs, distance_onehot
    from tt_bio.rf3.sampler import Draws
    from tt_bio.tenstorrent import get_device
    from perf.rf3.tt_rf3_bench import net_config

    cfg = net_config(args.ckpt)
    device = get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=args.num_steps, with_confidence=True)
    head = tt.confidence_head
    real_gln = ch.global_layer_norm

    report = {}
    for aa in args.aa:
        fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{aa}.json"),
                       n_recycles=max(args.n_recycles, 2), diffusion_batch_size=1,
                       seed=args.seed)[0]
        f, rep_atom_idxs = fo["feats"], fo["ground_truth"]["rep_atom_idxs"]
        host = HostInputs.build(f, device)
        s_inputs, s, z = tt.trunk(host, args.n_recycles)
        x_pred, _ = tt.sampler.sample(
            lambda x, t: tt.diffusion_module(host, x, t, s_inputs, s, z),
            torch.zeros(1, host.n_atom, 3), 1, draws=Draws())
        dist_tt = distance_onehot(x_pred[0:1], rep_atom_idxs, device)

        # --- 1. op level, both arms against fp64 on the same bf16 values -------------
        ops = {}
        exact_for: dict[tuple, torch.Tensor] = {}
        for name, t_tt in (("s_inputs", s_inputs), ("s_trunk", s), ("z_trunk", z)):
            xb = torch.Tensor(ttnn.to_torch(t_tt)).double()
            ref = exact_gln(xb)
            exact_for[tuple(t_tt.shape)] = ref
            got = {}
            for tag, fold in (("one_row_flatten", False), ("row_fold", True)):
                ch._GLN_ROW_FOLD = fold
                got[tag] = rel_rms(
                    torch.Tensor(ttnn.to_torch(real_gln(t_tt, kcfg))).float(), ref)
            ops[name] = {**got, "shape": list(t_tt.shape),
                         "ratio_fold_over_flatten":
                             round(got["row_fold"] / got["one_row_flatten"], 5)}

        # --- 2. head level. Device reference fed the exact normalisation -------------
        def exact_stub(x, compute_kernel_config):
            ref = exact_for[tuple(x.shape)]
            return ttnn.from_torch(ref.float(), dtype=x.dtype,
                                   layout=ttnn.TILE_LAYOUT, device=device)

        def run_head():
            out = head(s_inputs, s, z, dist_tt)
            got = {k: torch.Tensor(ttnn.to_torch(v)).float() for k, v in out.items()}
            for v in out.values():
                ttnn.deallocate(v)
            return got

        ch.global_layer_norm = exact_stub
        golden = run_head()
        ch.global_layer_norm = real_gln

        logits = {}
        for tag, fold in (("one_row_flatten", False), ("row_fold", True)):
            ch._GLN_ROW_FOLD = fold
            arm = run_head()
            logits[tag] = {k: round(rel_rms(arm[k], golden[k]), 8) for k in golden}

        # --- 3. time --------------------------------------------------------------
        def one(fold):
            ch._GLN_ROW_FOLD = fold
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = head(s_inputs, s, z, dist_tt)
            ttnn.synchronize_device(device)
            dt = time.perf_counter() - t0
            for v in out.values():
                ttnn.deallocate(v)
            return dt

        one(False)
        legs = {"A": [], "B": [], "A2": []}
        for _ in range(args.reps):
            for tag, fold in (("A", False), ("B", True), ("A2", False)):
                legs[tag].append(one(fold))

        def med(xs):
            xs = sorted(xs)
            return xs[len(xs) // 2]

        a, b, a2 = med(legs["A"]), med(legs["B"]), med(legs["A2"])
        report[str(aa)] = {
            "n_token": host.n_token,
            "op_rel_rms_vs_fp64": ops,
            "head_logits_rel_rms_vs_exact_gln": logits,
            "head_s": {"one_row_flatten": round(a, 5), "row_fold": round(b, 5),
                       "one_row_flatten_again": round(a2, 5)},
            "speedup": round(a / b, 4), "aa_floor": round(a / a2, 4),
        }
        print(json.dumps({str(aa): report[str(aa)]}, indent=2), flush=True)
        ch._GLN_ROW_FOLD = False

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
