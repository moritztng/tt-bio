#!/usr/bin/env python3
"""Per-leg parity and per-leg cost for the two atom-featurization legs, on REAL features.

The fold-level A/B cannot resolve this leg: at 76 aa the shipped arm's own run-to-run spread is
1.7 s on a 13 s fold, and the legs run ONCE per fold, so a longer fold dilutes them rather than
resolving them. So the leg is measured where it lives.

Features come from `build_openfold3_features` on the same `examples/ubq.yaml` the folds used, not
from a synthetic stand-in: the pair leg's `vlm` is `ref_space_uid` equality inside a block, so its
sparsity is a property of the real molecule and a random uid would price the wrong kernel.

Two readings per leg:
  parity  device against the host replica -- rel L2, PCC and max abs, on this cell's real inputs.
          The device module's own PCC gate (tests/test_openfold3_ref_atom_feat.py) is against the
          reference golden; this is against the thing it REPLACES, which is the comparison the
          inference A/B is made of.
  cost    host vs device wall clock over N repeats, AICLK read before and after.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


def aiclk(card):
    try:
        return int(Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read_text())
    except Exception:
        return None


def metrics(dev_t, host_t):
    import torch
    a = dev_t.double().reshape(-1)
    b = host_t.double().reshape(-1)
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    am, bm = a - a.mean(), b - b.mean()
    return {
        "rel_l2": float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-300)),
        "pcc": float((am * bm).sum() / (am.norm() * bm.norm() + 1e-300)),
        "max_abs": float((a - b).abs().max()),
        "host_norm": float(torch.linalg.vector_norm(b)),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="examples/ubq.yaml")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--card", default=os.environ.get("TT_VISIBLE_DEVICES", "1"))
    ap.add_argument("--out", default=str(Path(__file__).with_name("LEG_COST.json")))
    a = ap.parse_args()

    import json as _json
    import tempfile
    import torch
    import ttnn

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet)
    from tt_bio.openfold3_data import build_openfold3_features
    from tt_bio import openfold3_host_prep as HP
    from tt_bio.openfold3 import AtomEncoderTokenHead, RefAtomFeatureEmbedder
    from tt_bio.openfold3_weights import _sub
    from tt_bio.tenstorrent import get_device

    import yaml as _yaml
    # The featurizer wants OF3's own query shape, which `worker.py:1443` builds from the yaml.
    # One protein chain is all this cell has, so it is spelled out rather than imported out of a
    # 200-line CLI branch.
    spec = _yaml.safe_load(Path(a.yaml).read_text())
    chains = []
    for entry in spec["sequences"]:
        (mt, body), = entry.items()
        chains.append({"chain_ids": [body["id"]], "molecule_type": mt.upper(),
                       "sequence": body["sequence"], "smiles": None, "ccd_codes": None,
                       "main_msa_file_paths": None,
                       "paired_msa_file_paths": None, "template_alignment_file_path": None,
                       "template_entry_chain_ids": None, "sdf_file_path": None})
    stem = Path(a.yaml).stem
    query = {"query_name": stem, "use_msas": False, "use_paired_msas": False,
             "use_main_msas": False, "covalent_bonds": None, "chains": chains}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        _json.dump({"queries": {stem: query}}, fh)
        qpath = fh.name
    torch.manual_seed(0)
    import random as _pr
    _pr.seed(0)
    iqs = InferenceQuerySet.from_json(qpath)
    of3_query = next(iter(iqs.queries.values()))
    of3_query.use_msas = False
    of3_query.use_main_msas = False
    features = build_openfold3_features(of3_query, template_structures_directory=None)
    aux = HP.derive_block_aux(features)
    n_atom, NP, nb = aux["n_atom"], aux["NP"], aux["nb"]
    n_token = aux["n_token"]
    print(f"features: n_atom={n_atom} NP={NP} nb={nb} n_token={n_token}", flush=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    out = {
        "instrument": "of3t-hostleg leg_cost.py -- the two legs' device-vs-host parity and cost "
                      "on real ubq features",
        "host": f"qb1 (tt-quietbox) card {a.card}, Blackhole p150a",
        "cell": {"yaml": a.yaml, "n_atom": n_atom, "NP": NP, "nb": nb, "n_token": n_token},
        "checkpoint": CKPT,
        "repeats": a.repeats,
        "legs": {},
    }
    atom_mask = aux["atom_mask"]

    # ---- leg 1: the eight ref_atom_feature_embedder linears, both encoders ----
    for enc in ("diffusion_module.atom_attn_enc", "input_embedder.atom_attn_enc"):
        w = _sub(sd, f"{enc}.ref_atom_feature_embedder")
        cl_h, plm_h = HP.ref_atom_embed(w, features)
        rafe = RefAtomFeatureEmbedder(w, ckc)
        ins = HP.ref_atom_device_inputs(dev, features, atom_mask, NP)
        cl_d, plm_d = rafe(*ins)
        cl_dt = torch.Tensor(ttnn.to_torch(cl_d)).reshape(NP, 128)[:n_atom]
        plm_dt = torch.Tensor(ttnn.to_torch(plm_d)).reshape(plm_h.shape)
        cl_pad = torch.Tensor(ttnn.to_torch(cl_d)).reshape(NP, 128)[n_atom:]
        row = {
            "n_weights": len(w),
            "cl": metrics(cl_dt, cl_h),
            "plm": metrics(plm_dt, plm_h),
            "pad_rows_zero_on_device": bool((cl_pad.abs() < 1e-12).all()),
            "pad_rows_max_abs": float(cl_pad.abs().max()) if cl_pad.numel() else 0.0,
        }
        # cost, host then device, same inputs, warm
        HP.ref_atom_embed(w, features)
        clk0 = aiclk(a.card)
        t = []
        for _ in range(a.repeats):
            s = time.perf_counter(); HP.ref_atom_embed(w, features); t.append(time.perf_counter() - s)
        row["host_s"] = {"median": statistics.median(t), "min": min(t), "max": max(t)}
        rafe(*ins); ttnn.synchronize_device(dev)
        t = []
        for _ in range(a.repeats):
            s = time.perf_counter()
            c2, p2 = rafe(*ins)
            ttnn.synchronize_device(dev)
            t.append(time.perf_counter() - s)
            ttnn.deallocate(c2); ttnn.deallocate(p2)
        row["device_s"] = {"median": statistics.median(t), "min": min(t), "max": max(t)}
        row["aiclk_before"] = clk0
        row["aiclk_after"] = aiclk(a.card)
        row["speedup_host_over_device"] = row["host_s"]["median"] / row["device_s"]["median"]
        out["legs"][f"{enc}.ref_atom_feature_embedder"] = row
        print(enc, json.dumps(row), flush=True)
        for x in ins:
            ttnn.deallocate(x)
        ttnn.deallocate(cl_d); ttnn.deallocate(plm_d)

    # ---- leg 2: input_embedder linear_q.0 + the atom->token mean ----
    enc_sd = _sub(sd, "input_embedder.atom_attn_enc")
    lq_w = _sub(enc_sd, "linear_q")["0.weight"]
    a2t_mean = aux["atom_to_token_mean"]
    torch.manual_seed(7)
    ql = torch.randn(n_atom, 128) * 0.5           # the atom transformer's output, stand-in
    ai_h = (a2t_mean @ torch.nn.functional.linear(
        ql * atom_mask[:, None], lq_w.float()).relu())
    head = AtomEncoderTokenHead(_sub(enc_sd, "linear_q"), ckc)
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)
    ql_d = ft(ql.unsqueeze(0))
    amc = torch.zeros(1, n_atom, 1); amc[0, :, 0] = atom_mask
    amc_d, mean_d = ft(amc), ft(a2t_mean.unsqueeze(0))
    ai_d = head(ql_d, amc_d, mean_d)
    ai_dt = torch.Tensor(ttnn.to_torch(ai_d)).reshape(ai_h.shape)
    row = {"n_weights": 1, "note": "ql is a seeded stand-in for the atom transformer's output; "
                                   "the weight, the mask and the aggregation matrix are real",
           "ai": metrics(ai_dt, ai_h),
           "weight_shape": list(lq_w.shape)}
    clk0 = aiclk(a.card)
    t = []
    for _ in range(a.repeats):
        s = time.perf_counter()
        a2t_mean @ torch.nn.functional.linear(ql * atom_mask[:, None], lq_w.float()).relu()
        t.append(time.perf_counter() - s)
    row["host_s"] = {"median": statistics.median(t), "min": min(t), "max": max(t)}
    t = []
    for _ in range(a.repeats):
        s = time.perf_counter()
        o = head(ql_d, amc_d, mean_d)
        ttnn.synchronize_device(dev)
        t.append(time.perf_counter() - s)
        ttnn.deallocate(o)
    row["device_s"] = {"median": statistics.median(t), "min": min(t), "max": max(t)}
    row["aiclk_before"] = clk0
    row["aiclk_after"] = aiclk(a.card)
    row["speedup_host_over_device"] = row["host_s"]["median"] / row["device_s"]["median"]
    out["legs"]["input_embedder.atom_attn_enc.linear_q.0.weight"] = row
    print("linear_q head", json.dumps(row), flush=True)

    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
