#!/usr/bin/env python3
"""What the fused SDPA's compute config is worth on RF3 triangle attention, in error and in time.

Arms on ONE captured triangle-attention call, all scored against a torch fp64 evaluation of the
SAME bf16 operands, so the input quantisation is common to every arm and only the kernel's own
error is left:

    materialised   `_fp32_softmax_attention`, what the port ships -- bf16 scores to DRAM, fp32
                   copy, bias add, softmax, cast back. Correct and five DRAM traversals wide.
    fused_bf16     the fused SDPA at its op-default config, which is what "turn fp32_softmax off"
                   measured in pass 2 and what the 1.75e-2 z rel_rms there is really about.
    --sweep ckc    the same kernel across fidelity x math_approx x fp32_dest_acc.

The reference is `softmax(q@k^T * scale + bias * bias_scale) @ v`. The kernel applies `scale` after
the mask add (compute_common.hpp fuses it into the exp), so the two scales have to agree for the
fused arm to stand in at all -- `scale_pair_bias=True` bakes sqrt(head_dim) into the bias weight
and makes them agree, and the harness asserts it rather than assuming it.
"""
from __future__ import annotations

import argparse
import enum
import json
import statistics
import sys
import time
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rel_rms(got, want):
    return float(((got - want) ** 2).mean().sqrt() / (want ** 2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--calls", type=int, default=4, help="back-to-back calls per timed leg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--site", choices=("pairformer", "other"), default="pairformer",
                    help="pairformer: the 48-block stack, whose bias is pre-baked by "
                         "sqrt(head_dim). other: the MSA module and template embedder, which "
                         "pass scale_pair_bias=False, so the fused arm has to pre-bake it itself.")
    ap.add_argument("--sweep", choices=("ckc",), default=None,
                    help="walk fidelity x math_approx x fp32_dest_acc instead of the one config")
    ap.add_argument("--which", choices=("first", "last"), default="last",
                    help="which triangle-attention call of the recycle to capture")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=2, diffusion_batch_size=1, seed=args.seed)[0]
    f = fo["feats"]

    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from perf.rf3.tt_rf3_bench import net_config

    cfg = net_config(args.ckpt)
    device = T.get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=50, with_confidence=False)

    # Capture the first triangle-attention call the 48-block stack makes, operands and all.
    grab = {}
    orig = T._fp32_softmax_attention

    seen = {}

    def spy(q, k, v, bias, scale_inv, compute_kernel_config, out_dtype=ttnn.bfloat16,
            bias_scale_inv=None):
        # Triangle attention is the only caller whose batch dim IS the sequence dim: it runs one
        # attention per row of the pair tensor. The atom-level AttentionPairBias in the encoder is
        # [1, windows, 32, 32] and does not match, which is what the first run of this harness
        # captured by mistake.
        shp = tuple(int(d) for d in q.shape)
        # Triangle attention is the only caller whose batch dim IS the sequence dim. The extra
        # scale test picks out the 48-block Pairformer specifically: the template embedder and the
        # MSA module build their pair bias with scale_pair_bias=False, so their two scales differ
        # and the fused kernel cannot stand in for them at all.
        matched = abs(scale_inv - (bias_scale_inv if bias_scale_inv else scale_inv)) < 1e-12
        want = matched if args.site == "pairformer" else not matched
        if len(shp) == 4 and shp[0] == shp[2] and shp[0] > 1 and want:
            seen[shp] = seen.get(shp, 0) + 1
            if not grab or args.which == "last":
                grab.clear()
                grab.update(q=ttnn.to_torch(q), k=ttnn.to_torch(k), v=ttnn.to_torch(v),
                            bias=ttnn.to_torch(bias), scale_inv=scale_inv,
                            bias_scale_inv=bias_scale_inv, index=seen[shp])
        return orig(q, k, v, bias, scale_inv, compute_kernel_config, out_dtype, bias_scale_inv)

    T._fp32_softmax_attention = spy
    host = HostInputs.build(f, device)
    s_inputs, s_init, z_init = tt.feature_initializer(
        host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
        host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
        host.token_feats, host.relpos_feat, host.bond_feat)
    tmpl = tt.recycler.template_embedder.embed_template_feats(host.template_feats)
    s = ttnn.mul(s_init, 0.0)
    z = ttnn.mul(z_init, 0.0)
    tt.recycler(host, tmpl, host.msa_stack[0], s_inputs, s_init, z_init, s, z)
    T._fp32_softmax_attention = orig
    assert grab, "no triangle-attention call captured"

    qh, kh, vh, bh = grab["q"], grab["k"], grab["v"], grab["bias"]
    scale_inv, bias_scale_inv = grab["scale_inv"], grab["bias_scale_inv"]
    print(f"captured call {grab['index']} of {seen} -- q{list(qh.shape)} bias{list(bh.shape)} "
          f"scale_inv={scale_inv:.6f} bias_scale_inv={bias_scale_inv:.6f}", flush=True)
    # The fused kernel adds the mask before applying `scale`. Where the two scales already agree
    # (scale_pair_bias=True pre-baked sqrt(head_dim) into the bias weight) it can stand in as is;
    # where they do not, the fused arm multiplies the bias itself, which is what `bias_mul` is.
    bias_mul = bias_scale_inv / scale_inv
    print(f"  bias pre-multiply for the fused arm: {bias_mul:.6f}", flush=True)

    # The reference: the SAME bf16 operands, evaluated in fp64. Chunked over the batch so a
    # [512, 4, 512, 512] fp64 score tensor never exists.
    q64, k64, v64 = qh.double(), kh.double(), vh.double()
    b64 = bh.double()
    ref = torch.empty(qh.shape, dtype=torch.float64)
    for i in range(0, q64.shape[0], 8):
        sc = q64[i:i + 8] @ k64[i:i + 8].transpose(-1, -2)
        sc = sc * scale_inv + b64 * bias_scale_inv
        ref[i:i + 8] = torch.softmax(sc, dim=-1) @ v64[i:i + 8]
        del sc
    del q64, k64, v64, b64

    # The bf16 ceiling: the same maths in torch bf16 throughout, so "how close can bf16 storage of
    # the output even be" is a measured number and not a guess.
    ceil = torch.empty(qh.shape, dtype=torch.bfloat16)
    for i in range(0, qh.shape[0], 8):
        sc = (qh[i:i + 8].float() @ kh[i:i + 8].float().transpose(-1, -2))
        sc = sc * scale_inv + bh.float() * bias_scale_inv
        ceil[i:i + 8] = (torch.softmax(sc, dim=-1) @ vh[i:i + 8].float()).bfloat16()
        del sc

    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    qd, kd, vd, bd = up(qh), up(kh), up(vh), up(bh)
    # the fused arm's own bias, scaled on device exactly as the model scales it
    bd_f = bd if abs(bias_mul - 1.0) < 1e-12 else ttnn.multiply(bd, bias_mul)

    def run_materialised():
        return T._fp32_softmax_attention(qd, kd, vd, bd, scale_inv=scale_inv,
                                         compute_kernel_config=kcfg, out_dtype=ttnn.bfloat16,
                                         bias_scale_inv=bias_scale_inv)

    def run_fused_bf16():
        return T._tri_att_sdpa(qd, kd, vd, bd_f, scale_inv)

    def fused(ckc):
        def run():
            T._TRIATT_FUSED_HIFI_CKC = ckc
            return T._tri_att_sdpa_hifi(qd, kd, vd, bd_f, scale_inv)
        return run

    HF = {"HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4,
          "LoFi": ttnn.MathFidelity.LoFi}
    arms = [("materialised", run_materialised), ("fused_bf16", run_fused_bf16)]
    if args.sweep:
        # (fidelity, math_approx, fp32_dest_acc): the three terms the transcription threads into
        # the compute descriptor. dst_full_sync is left off throughout.
        for fid in ("LoFi", "HiFi2", "HiFi4"):
            for approx in (True, False):
                for acc in (False, True):
                    arms.append((f"{fid}_ap{int(approx)}_acc{int(acc)}",
                                 fused((HF[fid], approx, acc, False))))
    else:
        arms.append(("fused_hifi", fused(T._TRIATT_FUSED_HIFI_CKC)))

    rows = {}
    for name, fn in arms:
        try:
            o = fn()
        except Exception as exc:  # noqa: BLE001
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        if o is None:
            rows[name] = {"error": "declined"}
            print(f"  {name}: declined", flush=True)
            continue
        got = ttnn.to_torch(o).double()
        ttnn.deallocate(o)
        rows[name] = {"pcc": round(pcc(got, ref), 8), "rel_rms": rel_rms(got, ref)}
        print(f"  {name}: pcc {rows[name]['pcc']:.8f}  rel_rms {rows[name]['rel_rms']:.6e}",
              flush=True)
        del got
    rows["bf16_ceiling"] = {"pcc": round(pcc(ceil.double(), ref), 8),
                            "rel_rms": rel_rms(ceil.double(), ref)}
    print(f"  bf16 ceiling: pcc {rows['bf16_ceiling']['pcc']:.8f}  "
          f"rel_rms {rows['bf16_ceiling']['rel_rms']:.6e}", flush=True)

    # Timing: `--calls` back-to-back between two syncs, so per-call dispatch amortises the way it
    # does in a fold rather than being oversynced (tt-bio-isolated-op-timing-oversync-inflates-cost).
    times = {}
    for name, fn in arms:
        if "error" in rows.get(name, {}):
            continue
        legs = []
        for rep in range(args.reps + 1):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            outs = [fn() for _ in range(args.calls)]
            ttnn.synchronize_device(device)
            dt = (time.perf_counter() - t0) / args.calls
            for o in outs:
                ttnn.deallocate(o)
            if rep:
                legs.append(dt)
        times[name] = statistics.median(legs)
        print(f"  {name}: {times[name] * 1e3:8.3f} ms/call", flush=True)

    rep = {"aa": args.aa, "n_token": int(host.n_token), "shape": [int(d) for d in qh.shape],
           "accuracy": rows, "ms_per_call": {k: round(v * 1e3, 4) for k, v in times.items()},
           "speedup_vs_materialised": {
               k: round(times["materialised"] / v, 4) for k, v in times.items()}
           if "materialised" in times else {},
           "captured_call_index": grab["index"], "calls_seen": {str(k): v for k, v in seen.items()},
           "fused_hifi_stats": dict(T.TRIATT_FUSED_HIFI_STATS),
           "triatt_kernel_stats": {"served": T._triatt_sdpa.STATS[0],
                                   "declined": T._triatt_sdpa.STATS[1]},
           "rejects": {str(k): v for k, v in T._triatt_sdpa.REJECTS.items()}}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
