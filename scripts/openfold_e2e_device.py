"""OpenFold AF2 end-to-end with the DEVICE Evoformer trunk: the vendored reference runs
on CPU (embedders, extra-MSA, structure module, heads) EXCEPT the main Evoformer trunk,
which is swapped for the device tt_bio.openfold.EvoformerStack (real weights via
openfold_weights). 7ROA, single-seq, real weights. Reports device Ca-RMSD vs ground
truth — should track the host reference baseline (~13 A no-MSA), validating the device
port end-to-end."""
import time
import numpy as np
import torch
import ttnn

from tt_bio._vendor.openfold.config import model_config
from tt_bio._vendor.openfold.model.model import AlphaFold
from tt_bio._vendor.openfold.utils.import_weights import convert_deprecated_v1_keys
from tt_bio._vendor.openfold.data import data_pipeline, parsers, feature_pipeline
from tt_bio._vendor.openfold.data.templates import empty_template_feats
from tt_bio.openfold import EvoformerStack
from tt_bio.openfold_weights import evoformer_stack_subs
from tt_bio.tenstorrent import get_device

CKPT = "/home/ttuser/openfold_ckpt/finetuning_ptm_1.pt"
SEQ = ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISIS"
       "AIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG")
GT_CIF = "examples/ground_truth_structures/prot.cif"
C_S, HD_PAIR, H_PAIR, HD_MSA, H_MSA, NB = 384, 32, 4, 32, 8, 48


def gt_ca(path):
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
    return float(np.sqrt(((Pc @ R - Qc) ** 2).sum(1).mean()))


class _DeviceEvoformer(torch.nn.Module):
    """Adapter: reference evoformer forward signature -> device EvoformerStack."""
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, m, z, msa_mask, pair_mask, chunk_size=None, **kw):
        dev = get_device()
        ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        mo, zo, so = self.stack(ft(m), ft(z))
        tt = lambda t, shp: torch.as_tensor(ttnn.to_torch(t)).float().reshape(shp)
        s_shape = (m.shape[0], m.shape[2], C_S)
        return tt(mo, m.shape), tt(zo, z.shape), tt(so, s_shape)


def main():
    n = len(SEQ)
    cfg = model_config("model_1_ptm", train=False, low_prec=False)
    cfg.data.common.max_recycling_iters = 1
    model = AlphaFold(cfg).eval()
    sd = convert_deprecated_v1_keys(torch.load(CKPT, map_location="cpu", weights_only=False))
    model.load_state_dict(sd, strict=False)

    # build device trunk from the loaded evoformer weights, swap it in
    kcfg = ttnn.init_device_compute_kernel_config(
        get_device().arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    subs, s_lin = evoformer_stack_subs(dict(model.evoformer.state_dict()), NB)
    stack = EvoformerStack(subs, s_lin, HD_PAIR, H_PAIR, HD_MSA, H_MSA, kcfg)
    model.evoformer = _DeviceEvoformer(stack)

    seq_f = data_pipeline.make_sequence_features(SEQ, "target", n)
    msa = parsers.Msa(sequences=[SEQ], deletion_matrix=[[0] * n], descriptions=["query"])
    msa_f = data_pipeline.make_msa_features([msa])
    raw = {**seq_f, **msa_f, **empty_template_feats(n)}
    feats = feature_pipeline.FeaturePipeline(cfg.data).process_features(raw, mode="predict")
    feats = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}

    t0 = time.time()
    with torch.no_grad():
        out = model(feats)
    dt = time.time() - t0

    pos = out["final_atom_positions"]
    while pos.dim() > 3:
        pos = pos[0]
    ca = pos[:, 1, :].float().numpy()
    plddt = float(out["plddt"].mean()) if "plddt" in out else float("nan")
    gt = gt_ca(GT_CIF)
    pairs = [(ca[k - 1], xyz) for k, xyz in gt.items() if 1 <= k <= len(ca)]
    P = np.array([p for p, _ in pairs]); Q = np.array([q for _, q in pairs])
    rmsd = kabsch_rmsd(P, Q)
    print(f"pred Ca: {ca.shape[0]}  matched: {len(pairs)}  pLDDT: {plddt:.1f}  time: {dt:.1f}s")
    print(f"[OpenFold e2e | 7ROA | single-seq, DEVICE trunk, real weights] Ca-RMSD = {rmsd:.3f} A")


if __name__ == "__main__":
    main()
