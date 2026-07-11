"""Reference OpenFold AF2 end-to-end on CPU with REAL weights (finetuning_ptm_1.pt),
single-sequence (self-MSA, no templates), on the release-gate target 7ROA. Produces a
structure and reports Kabsch-aligned Cα-RMSD vs ground truth. This is the accuracy
baseline the device-trunk port must match (validates the vendored reference + weights).
Single-sequence + low recycling -> modest accuracy; the point is a real, honest RMSD
through the full pipeline.
"""
import sys, time
import numpy as np
import torch

from tt_bio._vendor.openfold.config import model_config
from tt_bio._vendor.openfold.model.model import AlphaFold
from tt_bio._vendor.openfold.utils.import_weights import convert_deprecated_v1_keys
from tt_bio._vendor.openfold.data import data_pipeline, parsers, feature_pipeline
from tt_bio._vendor.openfold.data.templates import empty_template_feats

CKPT = "/home/ttuser/openfold_ckpt/finetuning_ptm_1.pt"
SEQ = ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISIS"
       "AIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG")
GT_CIF = "examples/ground_truth_structures/prot.cif"


def gt_ca(path):
    # _atom_site cols: label_atom_id idx3, label_seq_id idx8, Cartn_x/y/z idx10/11/12.
    # Return {seq_id (1-indexed): xyz}; the gt starts mid-sequence, so align by seq_id.
    out = {}
    for ln in open(path):
        if ln.startswith("ATOM"):
            f = ln.split()
            if f[3] == "CA":
                out[int(f[8])] = (float(f[10]), float(f[11]), float(f[12]))
    return out


def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    Pr = Pc @ R
    return float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean()))


def main():
    n = len(SEQ)
    cfg = model_config("model_1_ptm", train=False, low_prec=False)
    cfg.data.common.max_recycling_iters = 1  # 2 iterations, for a faster first e2e
    model = AlphaFold(cfg).eval()
    sd = convert_deprecated_v1_keys(torch.load(CKPT, map_location="cpu", weights_only=False))
    model.load_state_dict(sd, strict=False)

    seq_f = data_pipeline.make_sequence_features(SEQ, "target", n)
    msa = parsers.Msa(sequences=[SEQ], deletion_matrix=[[0] * n], descriptions=["query"])
    msa_f = data_pipeline.make_msa_features([msa])
    raw = {**seq_f, **msa_f, **empty_template_feats(n)}
    fp = feature_pipeline.FeaturePipeline(cfg.data)
    feats = fp.process_features(raw, mode="predict")
    feats = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}

    t0 = time.time()
    with torch.no_grad():
        out = model(feats)
    dt = time.time() - t0

    pos = out["final_atom_positions"]  # [*, N, 37, 3]
    while pos.dim() > 3:
        pos = pos[0]
    ca = pos[:, 1, :].float().numpy()  # atom37 index 1 = CA
    plddt = float(out["plddt"].mean()) if "plddt" in out else float("nan")
    gt = gt_ca(GT_CIF)  # {seq_id: xyz}, seq_id 1-indexed into SEQ
    pairs = [(ca[k - 1], xyz) for k, xyz in gt.items() if 1 <= k <= len(ca)]
    P = np.array([p for p, _ in pairs]); Q = np.array([q for _, q in pairs])
    print(f"pred Ca: {ca.shape[0]}  gt Ca matched: {len(pairs)}  mean pLDDT: {plddt:.1f}  fold time: {dt:.1f}s")
    rmsd = kabsch_rmsd(P, Q)
    print(f"[OpenFold e2e | 7ROA | single-seq, ref, real weights] Ca-RMSD = {rmsd:.3f} A  (over {len(pairs)} residues)")


if __name__ == "__main__":
    main()
