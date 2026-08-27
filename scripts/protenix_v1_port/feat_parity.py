#!/usr/bin/env python3
"""Score tt_bio.protenix_data against upstream v0.5.0's own data pipeline.

The module PCCs (tt_parity.py) feed BOTH arms one feature dict, so they deliberately cannot see
a featurizer difference. This is the other half: same target through upstream's
`get_inference_dataloader` and through `tt_bio.protenix_data.build_complex_features`, compared
feature by feature.

Token count is the load-bearing one. A substituted CCD silently drops residues that fail lookup
and the only symptom is a smaller token count -- a 2024-06 CCD against a later one lost 61 of
the PD-L1 target's 116 residues. So the pinned components.v20240608.cif is what
PROTENIX_DATA_ROOT_DIR must point at, and the count is asserted, not eyeballed.

    PROTENIX_DATA_ROOT_DIR=/home/moritz/.coworker/protenix-ref-data \\
      ~/protenix05_ref_venv/bin/python scripts/protenix_v1_port/feat_parity.py \\
        --feats /tmp/pv1/feats_multimer.pt --seqs <A> <B>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

os.environ.setdefault("LAYERNORM_TYPE", "torch_layernorm")
os.environ.setdefault("PROTENIX_DATA_ROOT_DIR", "/home/moritz/.coworker/protenix-ref-data")

SRC = os.environ.get("PROTENIX_V050_SRC",
                     "/home/moritz/.coworker/protenix-ref-data/src/protenix-0.5.0")


def upstream_features(seqs, name="parity"):
    import torch
    sys.path.insert(0, SRC)
    os.chdir(SRC)
    from configs.configs_base import configs as configs_base
    from configs.configs_data import (CCD_COMPONENTS_FILE_PATH, data_configs)
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    print("CCD in use:", CCD_COMPONENTS_FILE_PATH, flush=True)
    assert "v20240608" in CCD_COMPONENTS_FILE_PATH, (
        "upstream resolved a CCD that is not the pinned components.v20240608.cif: "
        + CCD_COMPONENTS_FILE_PATH)

    spec = [{"sequences": [{"proteinChain": {"sequence": s, "count": 1}} for s in seqs],
             "name": name}]
    d = tempfile.mkdtemp(prefix="ptxfeat-")
    jp = os.path.join(d, "in.json")
    with open(jp, "w") as f:
        json.dump(spec, f)

    base = {**configs_base, **{"data": data_configs}, **inference_configs}
    base["input_json_path"] = jp
    base["dump_dir"] = d
    base["use_msa"] = False
    cfg = parse_configs(base, arg_str=[], fill_required_with_null=True)
    from protenix.data.infer_data_pipeline import get_inference_dataloader
    dl = get_inference_dataloader(configs=cfg)
    for batch in dl:
        data, _atom_array, err = batch[0]
        if err:
            raise RuntimeError(err)
        return data["input_feature_dict"], data
    raise RuntimeError("dataloader produced nothing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    ours = torch.load(args.feats, map_location="cpu", weights_only=False)["feats"]
    # Upstream is featurized TWICE. Some of its features are not reproducible -- `ref_pos` is a
    # stochastic RDKit conformer, and two upstream runs of the identical input differ by ~9 A --
    # so a single draw is not a reference for those. Scoring against upstream's own run-to-run
    # spread separates "the port disagrees" from "upstream does not agree with itself", and it
    # does so for whatever feature happens to be stochastic rather than for a hardcoded name.
    theirs, meta = upstream_features(args.seqs, name="ref0")
    theirs2, _ = upstream_features(args.seqs, name="ref1")

    n_ours = int(ours["residue_index"].shape[-1])
    n_theirs = int(theirs["residue_index"].shape[-1])
    a_ours = int(ours["ref_pos"].shape[0])
    a_theirs = int(theirs["ref_pos"].shape[-2])
    print("\nN_token  ours=%d  upstream=%d" % (n_ours, n_theirs))
    print("N_atom   ours=%d  upstream=%d" % (a_ours, a_theirs))
    seqlen = sum(len(s) for s in args.seqs)
    print("sum(len(seq)) = %d" % seqlen)

    rows, npass = [], 0
    def check(name, ok, detail=""):
        nonlocal npass
        npass += bool(ok)
        rows.append((name, ok, detail))

    check("N_token == sum of sequence lengths (no CCD residue dropped)",
          n_ours == seqlen, "%d vs %d" % (n_ours, seqlen))
    check("N_token matches upstream", n_ours == n_theirs, "%d vs %d" % (n_ours, n_theirs))
    check("N_atom matches upstream", a_ours == a_theirs, "%d vs %d" % (a_ours, a_theirs))

    shared = sorted(set(ours) & set(theirs))
    print("\nshared feature keys: %d   ours-only: %s   upstream-only: %s"
          % (len(shared), sorted(set(ours) - set(theirs)), sorted(set(theirs) - set(ours))))
    print("\n%-26s %-18s %-18s %s" % ("feature", "ours", "upstream", "verdict"))
    for k in shared:
        a, b = ours[k], theirs[k]
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            continue
        a = a.squeeze(); b = b.squeeze()
        same_shape = tuple(a.shape) == tuple(b.shape)
        if same_shape:
            md = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
            eq = md == 0.0
            # upstream's own spread on this feature, from its second draw
            c = theirs2.get(k)
            self_d = (float((b.double() - c.squeeze().double()).abs().max())
                      if torch.is_tensor(c) and tuple(c.squeeze().shape) == tuple(b.shape)
                      else 0.0)
            if eq:
                v, ok = "EXACT", True
            elif md < 1e-5:
                v, ok = "max|d|=%.3g" % md, True
            elif self_d > 0 and md <= self_d * 1.5:
                v = "STOCHASTIC ours-vs-ref %.3g <= upstream self %.3g" % (md, self_d)
                ok = True
            else:
                v, ok = "max|d|=%.3g (upstream self %.3g)" % (md, self_d), False
        else:
            v = "SHAPE"
            ok = False
        check("feature " + k, ok, v)
        print("%-26s %-18s %-18s %s" % (k, tuple(a.shape), tuple(b.shape), v))

    print("\n%-70s %s" % ("check", "verdict"))
    for name, ok, detail in rows:
        print("%-70s %-5s %s" % (name, "PASS" if ok else "FAIL", detail))
    print("\nFEATURIZER %d/%d" % (npass, len(rows)))
    return 0 if npass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
