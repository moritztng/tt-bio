#!/usr/bin/env python3
"""ESMFold2 on CPU in fp32, no Tenstorrent device: the float reference for the SC26 GFP question.

    ESM_ROOT=/home/ttuser/esm python perf/sc26_gfp/cpu_reference.py --seq <aa> --seeds 0,1,2 \
        --out runs/gfp_ref.json [--steps 20] [--loops 3] [--dtype fp32|fp64]

The folding model is the vendored ESMFold2Model left unpatched (the torch reference every port
leg is scored against). Its language model is the esm-repo ESMC built at the 6B config with the
pinned ESMC-6B snapshot in fp32 (scripts/esmc_embed_parity.py's reference), so nothing here
touches ttnn. The run mirrors demo/sc26/engine/chipworker.py: one chain, no MSA, 20 steps,
3 recycles, one sample, the seed inside _seed_context. Per seed it writes mean pLDDT, pTM and the
C-alpha coordinates, which score.py compares to the crystal.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]


class CpuESMC:
    """The `_ESMCAdapter` contract on CPU: hidden states [n_layers+1, B, L, d], hs[0] the embedding,
    hs[i] block i-1's output, hs[-1] the final-norm output (tt_bio.esmc.ESMCHiddenStatesModel)."""

    def __init__(self, dtype):
        from esmc_embed_parity import load_reference, reference_state_dict
        self.lm = load_reference("esmc-6b", reference_state_dict("esmc-6b")).to(dtype).eval()
        self.dtype = dtype

    def __call__(self, input_ids, sequence_id=None, output_hidden_states=True, **_):
        import types
        if sequence_id is not None:
            assert bool((sequence_id >= 0).all()), "padding is not handled by this reference"
        x = self.lm.embed(input_ids)
        post, _pre, hidden, _ = self.lm.transformer(x, sequence_id)
        hs = torch.stack([x, *hidden[:-1], post], 0).float()
        return types.SimpleNamespace(hidden_states=hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp64"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    dtype = {"fp32": torch.float32, "fp64": torch.float64}[args.dtype]

    from tt_bio._vendor.esm.models.esmfold2 import ESMFold2InputBuilder
    from tt_bio._vendor.esm.models.esmfold2.processor import _seed_context
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model
    from tt_bio.esmfold2_runtime import build_spi
    from tt_bio.weights import ESMFOLD2_REPO, hf_revision

    t0 = time.time()
    rev = hf_revision(ESMFOLD2_REPO)
    model = ESMFold2Model.from_pretrained(ESMFOLD2_REPO, load_esmc=False, revision=rev).to(dtype).eval()
    model._esmc = CpuESMC(dtype)
    print(f"loaded in {time.time() - t0:.0f}s, ESMFold2 {ESMFOLD2_REPO}@{rev}, {args.dtype}", flush=True)
    builder = ESMFold2InputBuilder()
    seq = args.seq.strip().upper()
    out = {"sequence": seq, "steps": args.steps, "loops": args.loops, "dtype": args.dtype,
           "esmfold2_revision": rev, "runs": {}}
    if os.path.exists(args.out):
        out["runs"] = json.load(open(args.out)).get("runs", {})
    for seed in map(int, args.seeds.split(",")):
        if str(seed) in out["runs"]:
            continue
        t = time.time()
        feats, chain_infos = builder.prepare_input(build_spi([("A", seq)]), seed=seed, device="cpu")
        feats = {k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in feats.items()}
        with _seed_context(seed):
            o = model(**feats, num_loops=args.loops, num_sampling_steps=args.steps,
                      num_diffusion_samples=1, early_exit=False)
        res = builder.decode(o, feats, chain_infos, num_diffusion_samples=1)
        mask = feats["atom_attention_mask"][0].bool()
        names = ["".join(chr(int(c) + 32) for c in n if int(c)).strip()
                 for n in feats["ref_atom_name_chars"][0][mask]]
        resid = feats["atom_to_token"][0][mask].tolist()
        xyz = o["sample_atom_coords"][0][mask].double()
        ca = [xyz[i].tolist() for i, nm in enumerate(names) if nm == "CA"]
        assert len(ca) == len(seq), (len(ca), len(seq))
        plddt = res.plddt.flatten().double()
        out["runs"][str(seed)] = dict(plddt_mean=float(plddt.mean()), plddt=plddt.tolist(),
                                      ptm=float(res.ptm) if res.ptm is not None else None,
                                      ca=ca, seconds=round(time.time() - t, 1))
        print(f"seed {seed}: mean pLDDT {float(plddt.mean()):.4f} pTM {res.ptm} "
              f"in {time.time() - t:.0f}s", flush=True)
        Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
