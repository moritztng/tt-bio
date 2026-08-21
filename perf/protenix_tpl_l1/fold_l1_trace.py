"""What L1 actually looks like at the template embedder's residency window, in a real fold.

The isolated probe (probe.py) replays the window on an idle device. This runs the real
protenix-v2 trunk and reads the allocator at the moment `_template` holds the normed pair
tensor in L1, so the margin is measured against everything else the fold has live -- which
is the term the boltz2 crash actually died on (247 KB/core of other live L1 buffers).

    TT_VISIBLE_DEVICES=0 python3 perf/protenix_tpl_l1/fold_l1_trace.py --n 496

Prints one JSON line per template projection, then a summary line.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

# A CDK2 tandem repeat, truncated to the requested length: token count == residue count for a
# single protein chain, so `--n` IS the pair tensor's side.
SEQ = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQ"
       "DLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRA"
       "PEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPL"
       "DEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="residues (= tokens = pair side)")
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--steps", type=int, default=8, help="diffusion steps (this site is trunk-only)")
    ap.add_argument("--reserve", type=int, default=None,
                    help="override _PAIR_L1_CONSUMER_RESERVE for the template site")
    ap.add_argument("--headroom", type=float, default=None)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio import protenix as P
    from tt_bio import tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts

    dev = get_device()

    def free_per_bank():
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
        return int(mv.total_bytes_free_per_bank)

    trace: list[dict] = []
    state_l1 = {"in_l1": False, "n": 0}
    _orig_ln, _orig_proj = P._l1_layer_norm, P._narrow_proj_linear

    def ln(x, headroom, reserve_per_core=0, **kw):
        if args.headroom is not None:
            headroom = args.headroom
        if args.reserve is not None:
            reserve_per_core = args.reserve
        before = free_per_bank()
        out, in_l1 = _orig_ln(x, headroom, reserve_per_core, **kw)
        state_l1["in_l1"] = bool(in_l1)
        state_l1["n"] = 0
        trace.append({"ev": "ln", "shape": list(x.shape), "headroom": headroom,
                      "reserve": reserve_per_core, "in_l1": bool(in_l1),
                      "free_before": before, "free_after": free_per_bank()})
        return out, in_l1

    def proj(x, w, ckc, dtype, l1_out=False):
        if not l1_out:
            return _orig_proj(x, w, ckc, dtype, l1_out=l1_out)
        before = free_per_bank()
        refused = len(T._L1_OUT_REFUSED)
        out = _orig_proj(x, w, ckc, dtype, l1_out=l1_out)
        state_l1["n"] += 1
        if state_l1["n"] <= 2:      # first two of the nt projections per cycle
            trace.append({"ev": "proj", "x": list(x.shape), "w": list(w.shape),
                          "free_before": before, "free_after": free_per_bank(),
                          "out": None if out is None else
                                 str(out.memory_config().buffer_type).split(".")[-1],
                          "newly_refused": len(T._L1_OUT_REFUSED) - refused})
        return out

    P._l1_layer_norm, P._narrow_proj_linear = ln, proj

    seq = (SEQ * (args.n // len(SEQ) + 1))[:args.n]
    work = Path(tempfile.mkdtemp(prefix="tplL1-"))
    cfg = dict(model="protenix-v2", fast=False, output_format="cif",
               recycling_steps=args.cycles, sampling_steps=args.steps, diffusion_samples=1,
               seed=0, trace=False, msa_dir=str(work / "msa"), struct_dir=str(work / "out"),
               use_msa_server=False, msa_db_path=None, use_envdb=False, msa_endpoint=None,
               single_sequence=True, msa_server_url=None, msa_pairing_strategy="greedy",
               msa_server_username=None, msa_server_password=None, api_key_value=None,
               max_msa_seqs=8192, write_pae=False, write_pde=False, write_embeddings=False,
               method=None)
    _ensure_local_artifacts(cfg)
    st = _WorkerState("tenstorrent")
    st.load_model(cfg)

    from tt_bio.protenix_data import build_complex_features
    feats = build_complex_features([(seq, None, "protein")], mol_dir=cfg.get("mol_dir"),
                                   chain_ids=["A"], bonds=None)
    n_tok = int(feats["restype"].shape[0])
    print(json.dumps({"grid": list(T.COMPUTE_GRID_MAIN), "n_res": args.n, "n_tokens": n_tok,
                      "free_per_bank_idle": free_per_bank(),
                      "banks": int(ttnn.get_memory_view(dev, ttnn.BufferType.L1).num_banks)}),
          flush=True)
    _, conf = st.model.fold(dict(feats), n_step=args.steps, n_sample=1, seed=0,
                            progress_fn=lambda *a, **k: None, return_confidence=True,
                            n_cycles=args.cycles, max_parallel_samples=1, trace=False)
    for t in trace[:12]:
        print(json.dumps(t), flush=True)
    lns = [t for t in trace if t["ev"] == "ln"]
    projs = [t for t in trace if t["ev"] == "proj"]
    c = conf[0] if isinstance(conf, list) else conf
    print(json.dumps({"summary": True, "n_tokens": n_tok, "ln_calls": len(lns),
                      "ln_in_l1": sum(t["in_l1"] for t in lns),
                      "min_free_after_zn": min((t["free_after"] for t in lns), default=None),
                      "min_free_after_proj": min((t["free_after"] for t in projs), default=None),
                      "proj_dram_outs": sum(1 for t in projs if t["out"] == "DRAM"),
                      "refusals": sorted(str(k) for k in T._L1_OUT_REFUSED),
                      "plddt": float(c["plddt"])}), flush=True)


if __name__ == "__main__":
    main()
