"""Where the host template embedding's seconds go at 848 residues, and what the card would charge.

Pass 6 of the port put the template embedding on host and cleared it against a 1.0 s bar at 208
residues. Pass 5 of this program measured it at 15.894 s at 848, 9.0 % of the fold. This splits it
at the four seams `TemplateEmbedding` already has, so the fix can be aimed rather than guessed:

  prologue   the 88-channel pair build (distogram, two one-hots, the two masks) and embedding2d
  pair_stack the 2 PairBlocks at c_t=64 -- the only O(L^3) part, and the part `AF2PairBlock`
             already knows how to run on card (`evoformer_order=False`)
  norm       output_norm
  attn       the pointwise attention over templates, degenerate at one template

**The dtype is the production one.** `TemplateEmbedding.pair_representation` reads
`dtype = pair.dtype` and `AF2Model.forward` hands it `trunk_dtype`, which is bfloat16 -- so the
host template runs in torch bfloat16, not float32. Pass 6 of this program hardcoded float32 here
and its 36.509 s pair stack and 49.242 s whole forward are that arm, not the shipped one. `--dtype`
defaults to the model's own `trunk_dtype`; `--dtype float32` reproduces pass 6 and is also the
port's envelope arm.

`--device` adds the arm this program actually needs to price: the two `PairBlock`s as
`AF2PairBlock(scoped("template.pair_stack.N."), ckc, head_dim=16, n_heads=4,
evoformer_order=False)`, one upload of the prologue's output and one download, on card. Masks go
in as `None` for the same reason the trunk's do: `mask_2d` is all ones for every fold this port
serves. It is a COST measurement. The rms it prints against the host arm is a per-stack number on
one input and is NOT an accuracy verdict -- this stack enters `pair` upstream of all 52 trunk
blocks, so `af2ig-chained-error-accumulates-coherently-per-op-instrument-blind` applies with more
force here than anywhere it has already fired. The decision instruments are `tap_gate.py --device`
and `filter_flip_rate.py`.

Host arms are torch only. The inputs are the production featurisation of a synthetic two-chain
complex, same as fold_timing.py, so no arm here makes an accuracy claim.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


def prologue_act(template, feats, tokens, dtype, multichain_mask):
    """The 88-channel build plus `embedding2d`, i.e. the pair stack's input. Untimed."""
    from torch.nn import functional as F

    from tt_bio.af2_reference import ATOM_ORDER, TEMPLATE_DGRAM, dgram_from_positions

    pb_mask = feats["template_pseudo_beta_mask"][0].float()
    mask_pb = pb_mask[:, None] * pb_mask[None, :] * multichain_mask
    dgram = dgram_from_positions(feats["template_pseudo_beta"][0], *TEMPLATE_DGRAM)
    aatype = F.one_hot(feats["template_aatype"][0].long(), 22).to(dtype)
    atom_mask = feats["template_all_atom_mask"][0].float()
    bb = atom_mask[:, ATOM_ORDER["N"]] * atom_mask[:, ATOM_ORDER["CA"]] \
        * atom_mask[:, ATOM_ORDER["C"]]
    mask_bb = (bb[:, None] * bb[None, :] * multichain_mask).to(dtype)
    act = torch.cat([
        (dgram * mask_pb[..., None]).to(dtype),
        mask_pb.to(dtype)[..., None],
        aatype[None].expand(tokens, -1, -1),
        aatype[:, None].expand(-1, tokens, -1),
        torch.zeros(tokens, tokens, 3, dtype=dtype),
        mask_bb[..., None],
    ], dim=-1) * mask_bb[..., None]
    return template.embedding2d(act)


def device_arm(model, act, reps: int) -> dict:
    """The two template `PairBlock`s on card: upload, block 0, block 1, download.

    Every leg ends in `ttnn.synchronize_device`, which is what makes the per-block split honest and
    also slightly oversyncs the stack total against a fold that would not sync between blocks
    (`tt-bio-isolated-op-timing-oversync-inflates-cost`). Both are reported: `stack_s` is the sum of
    the two synced blocks, `seam_s` adds the round trip.
    """
    import ttnn

    from tt_bio.af2 import AF2PairBlock, compute_kernel_config, get_device

    state = model.state_dict()

    def scoped(prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

    ckc = compute_kernel_config()
    device = get_device()
    blocks = [AF2PairBlock(scoped(f"template.pair_stack.{i}."), ckc,
                           head_dim=16, n_heads=4, evoformer_order=False)
              for i in range(len(model.template.pair_stack))]
    grid = str(device.compute_with_storage_grid_size())
    up = act.unsqueeze(0).to(torch.bfloat16)

    rows = []
    out_torch = None
    for _ in range(reps):
        row = {}
        t0 = time.perf_counter()
        z = ttnn.from_torch(up, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        ttnn.synchronize_device(device)
        row["upload_s"] = time.perf_counter() - t0
        for index, block in enumerate(blocks):
            t0 = time.perf_counter()
            z = block(z)
            ttnn.synchronize_device(device)
            row[f"block{index}_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out_torch = torch.Tensor(ttnn.to_torch(z)).squeeze(0)
        row["download_s"] = time.perf_counter() - t0
        ttnn.deallocate(z)
        row["stack_s"] = sum(row[f"block{i}_s"] for i in range(len(blocks)))
        row["seam_s"] = row["stack_s"] + row["upload_s"] + row["download_s"]
        rows.append(row)
    warm = rows[1:] or rows
    keys = sorted(rows[0])
    return {"grid": grid, "blocks": len(blocks), "reps": reps, "rows": rows,
            "warm_median_s": {k: statistics.median([r[k] for r in warm]) for k in keys},
            "out_shape": list(out_torch.shape), "out_dtype": str(out_torch.dtype),
            "out_abs_mean": float(out_torch.float().abs().mean()),
            "_out": out_torch}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dtype", default="model", choices=["model", *DTYPES])
    ap.add_argument("--device", action="store_true", help="also price the pair stack on card")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from fold_timing import feats_from_pdb, to_torch
    from tap_gate import DEFAULT_PARAMS
    from tt_bio.af2_reference import load_af2_model
    from tt_bio.af2_weights import load_af2_state_dict

    feats_np, binder_seq = feats_from_pdb(args.pdb)
    feats = {k: to_torch(v) for k, v in feats_np.items()}

    model = load_af2_model(load_af2_state_dict(args.params or DEFAULT_PARAMS), template=True)
    model.eval()
    template = model.template

    tokens = int(feats["seq_mask"].shape[0])
    dtype = model.trunk_dtype if args.dtype == "model" else DTYPES[args.dtype]
    pair = torch.zeros(tokens, tokens, 128, dtype=dtype)
    mask_2d = (feats["seq_mask"][:, None] * feats["seq_mask"][None, :]).to(dtype)
    asym = feats["asym_id"]
    multichain_mask = (asym[:, None] == asym[None, :]).to(dtype)

    # The same seams `pair_representation` runs, timed one at a time. Single template, which is
    # what AF2-IG feeds it, so the template loop runs once.
    def legs() -> dict:
        out = {}
        t0 = time.perf_counter()
        act = prologue_act(template, feats, tokens, dtype, multichain_mask)
        out["prologue_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        for block in template.pair_stack:
            act = block(act, mask_2d)
        out["pair_stack_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        act = template.output_norm(act)
        out["norm_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        repr_ = torch.stack([act], dim=0)
        template_mask = feats["template_mask"].to(dtype)
        flat_query = pair.reshape(tokens * tokens, 1, 128)
        flat_templates = repr_.permute(1, 2, 0, 3).reshape(tokens * tokens, 1, 64)
        bias = 1e9 * (template_mask[None, None, None, :] - 1.0)
        emb = template.attn(flat_query, flat_templates, bias).reshape(tokens, tokens, 128)
        emb = emb * (template_mask.sum() > 0).to(emb.dtype)
        out["attn_s"] = time.perf_counter() - t0
        out["total_s"] = sum(out[k] for k in ("prologue_s", "pair_stack_s", "norm_s", "attn_s"))
        out["out_abs_mean"] = float(emb.float().abs().mean())
        return out

    report = {"mode": "af2ig_template_split", "tokens": tokens,
              "binder_residues": len(binder_seq), "dtype": str(dtype),
              "threads": torch.get_num_threads(), "reps": args.reps}
    with torch.no_grad():
        reps = [legs() for _ in range(args.reps)]
        whole = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            template(pair, feats, mask_2d, multichain_mask)
            whole.append(time.perf_counter() - t0)

        warm = reps[1:] or reps
        report["warm_median_s"] = {
            k: statistics.median([r[k] for r in warm])
            for k in ("prologue_s", "pair_stack_s", "norm_s", "attn_s", "total_s")}
        report["whole_forward_s"] = whole
        report["whole_forward_warm_median_s"] = statistics.median(whole[1:] or whole)
        report["rows"] = reps

        if args.device:
            host_act = prologue_act(template, feats, tokens, dtype, multichain_mask)
            host_out = host_act.clone()
            for block in template.pair_stack:
                host_out = block(host_out, mask_2d)
            dev = device_arm(model, host_act, args.reps)
            out_torch = dev.pop("_out")
            delta = (out_torch.float() - host_out.float())
            dev["rms_vs_host"] = float(delta.pow(2).mean().sqrt())
            dev["host_rms"] = float(host_out.float().pow(2).mean().sqrt())
            dev["host_out_abs_mean"] = float(host_out.float().abs().mean())
            dev["note"] = ("cost measurement; the rms is one input, one stack, NOT the decision "
                           "criterion -- see tap_gate.py --device and filter_flip_rate.py")
            report["device"] = dev

    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
