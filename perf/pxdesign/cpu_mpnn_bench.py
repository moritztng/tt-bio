"""Host-CPU cost of the ProteinMPNN sequence-design stage, at PXDesign's real shape.

PXDesign runs ProteinMPNN (SolubleMPNN is the same architecture with soluble-only
weights) as a fresh subprocess per invocation, designing a binder against a fixed
target. `sample` decodes autoregressively over EVERY node of the complex, so the
step count is target+binder, not binder alone.

Weights are random: the wall clock of a fixed-shape forward does not depend on their
values, and this avoids a checkpoint download. Architecture and hyperparameters match
`pxdbench/tools/protmpnn/protein_mpnn_run.py` (hidden 128, 3+3 layers, k from the
v_48_020 checkpoint = 48).
"""
import argparse, importlib.util, json, os, time

import torch

# Load the module by path: importing it as a package pulls pxdbench/tools/__init__,
# which imports protenix. protein_mpnn_utils.py itself only needs torch and numpy.
_SRC = os.environ.get("PXDBENCH_ROOT", os.path.expanduser("~/PXDesignBench"))
_spec = importlib.util.spec_from_file_location(
    "protein_mpnn_utils",
    os.path.join(_SRC, "pxdbench/tools/protmpnn/protein_mpnn_utils.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ProteinMPNN = _mod.ProteinMPNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, nargs="+", default=[116, 768])
    ap.add_argument("--binder", type=int, default=80)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-seqs", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    torch.manual_seed(0)
    rec = {"host": os.uname().nodename, "torch_threads": torch.get_num_threads(),
           "k": args.k, "hidden": args.hidden, "num_seqs": args.num_seqs, "cells": []}

    t0 = time.perf_counter()
    model = ProteinMPNN(num_letters=21, node_features=args.hidden,
                        edge_features=args.hidden, hidden_dim=args.hidden,
                        num_encoder_layers=3, num_decoder_layers=3,
                        k_neighbors=args.k, augment_eps=0.0)
    model.eval()
    rec["model_construct_s"] = time.perf_counter() - t0
    rec["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6

    for tgt in args.target:
        N = tgt + args.binder
        X = torch.randn(1, N, 4, 3) * 10.0
        S = torch.randint(0, 20, (1, N))
        mask = torch.ones(1, N)
        chain_mask = torch.zeros(1, N)
        chain_mask[:, tgt:] = 1.0                    # design the binder only
        chain_enc = torch.cat([torch.ones(1, tgt), 2 * torch.ones(1, args.binder)], -1)
        residue_idx = torch.cat([torch.arange(tgt).unsqueeze(0),
                                 torch.arange(args.binder).unsqueeze(0) + 100 + tgt], -1)
        chain_M_pos = torch.ones(1, N)
        omit = torch.zeros(21, dtype=torch.float32).numpy()

        ts = []
        with torch.no_grad():
            for r in range(args.reps + 1):
                randn = torch.randn(1, N)
                t0 = time.perf_counter()
                for _ in range(args.num_seqs):
                    model.sample(X, randn, S, chain_mask, chain_enc, residue_idx,
                                 mask=mask, temperature=0.0001, omit_AAs_np=omit,
                                 bias_AAs_np=torch.zeros(21).numpy(),
                                 chain_M_pos=chain_M_pos,
                                 bias_by_res=torch.zeros(1, N, 21))
                dt = time.perf_counter() - t0
                if r:                                 # rep 0 is cold
                    ts.append(dt)
        ts.sort()
        cell = {"target_residues": tgt, "binder_length": args.binder, "n_nodes": N,
                "decode_steps": N, "cold_s": None, "median_s": ts[len(ts) // 2],
                "min_s": ts[0], "max_s": ts[-1], "all_warm_s": ts}
        rec["cells"].append(cell)
        print("target %4d + binder %d = %4d nodes: median %.3f s  (min %.3f max %.3f)"
              % (tgt, args.binder, N, cell["median_s"], cell["min_s"], cell["max_s"]),
              flush=True)

    print("model construct %.2f s, %.2f M params, %d torch threads"
          % (rec["model_construct_s"], rec["params_M"], rec["torch_threads"]))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
