#!/usr/bin/env python3
# ODesign CTE isolation (pass-8 lever 4). Runs the on-device ConstraintTemplateEmbedder
# (full: distogram binning + linear_no_bias_a + linear_no_bias_z(LN_z(z)) + 2 pair-only
# PairformerLayers + LN_v + relu + linear_no_bias_u) on the captured z_init, and
# compares to ODesign's own ConstraintTemplateEmbedder run fresh in fp32 on the SAME
# z_init + constraint_feature. The CTE is the one new component vs protenix; if its
# isolated PCC is < 0.999, that's the z bug to chase. If it's clean, the 10-cycle z
# gap is the PF/MSA pair-accumulation bf16 floor (pass-8 levers 1-3).
import os, sys, pickle
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
os.environ.setdefault("DATA_ROOT_DIR", "./data")

GOLDEN = "/home/moritz/.coworker/scratch/odesign-ref/golden"
CKPT = "/home/moritz/.coworker/scratch/odesign-ref/ckpt/odesign_base_prot_flex.pt"
TRUNK_IN = os.path.join(GOLDEN, "odesign_trunk_inputs.pkl")


def _pcc(u, v):
    u = u.flatten().double(); v = v.flatten().double()
    return float(((u - u.mean()) * (v - v.mean())).sum()
                 / ((u - u.mean()).norm() * (v - v.mean()).norm() + 1e-12))


def _maxerr(u, v):
    return float((u.float() - v.float()).abs().max())


def cpu_cte(z_init, constraint_feature, sd):
    """ODesign's ConstraintTemplateEmbedder run fresh in fp32 on CPU."""
    import torch.nn.functional as F
    sys.path.insert(0, "/home/moritz/.coworker/scratch/odesign-ref/ODesign")
    from src.model.modules.pairformer import ConstraintTemplateEmbedder
    cte_sd = {k[len("constraint_distogram_embedder."):]: v for k, v in sd.items()
              if k.startswith("constraint_distogram_embedder.")}
    cte = ConstraintTemplateEmbedder(c_z=128)
    cte.load_state_dict(cte_sd, strict=False); cte.eval(); torch.set_grad_enabled(False)
    # ODesign CTE.forward(input_data, z): bins input_data.constraint_feature, adds linear_no_bias_z(LN_z(z))
    class _In:
        pass
    _in = _In()
    _in.constraint_feature = constraint_feature
    z = z_init.unsqueeze(0).float()  # (1,N,N,128)
    u = cte(_in, z)
    return u.squeeze(0).float()


def main():
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.odesign import ODesignTrunk
    fd = pickle.load(open(TRUNK_IN, "rb"))["feature_data"]
    feat = {k: fd[k] for k in fd}
    ck = torch.load(CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    trunk = ODesignTrunk(sd, ckc, dev)

    # build z_init on host via the pass-6 front-end (RPE + trunk-init linears)
    s_inputs = trunk.ti.input_feature_embedder(feat)
    relp = trunk.ti.rpe(feat)
    token_bonds = feat["token_bonds"].float()
    s_init, z_init = trunk.ti.trunk_init(s_inputs, relp, token_bonds)  # host (N,384),(N,N,128)
    cf = feat["constraint_feature"].float()
    N = z_init.shape[0]
    print("z_init", tuple(z_init.shape), "constraint_feature", tuple(cf.shape))

    # on-device CTE: a_term (precompute) + _cte(z_init_dev)
    a_term = trunk._ctet_a_term(cf)
    z_init_dev = ttnn.from_torch(z_init.float().reshape(1, N, N, trunk.C_Z),
                                 layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    u_dev = trunk._cte(z_init_dev, a_term)
    u_dev_h = torch.Tensor(ttnn.to_torch(u_dev)).float().reshape(N, N, trunk.C_Z)

    # CPU-fp32 reference CTE on the same z_init
    u_ref = cpu_cte(z_init, cf, sd)

    p = _pcc(u_dev_h, u_ref); m = _maxerr(u_dev_h, u_ref)
    verdict = "CLEAN" if p >= 0.999 else "CHECK"
    print("\n=== CTE isolated (device bf16 vs CPU-fp32, same z_init + constraint_feature) ===")
    print("  u_ij (N,N,128)  PCC %.6f  maxerr %.4e" % (p, m))
    if p >= 0.999:
        print("  -> CLEAN: CTE is parity-clean; the 10-cycle z gap is the PF/MSA pair-accumulation bf16 floor")
    else:
        print("  -> CHECK: CTE has a precision issue to chase (pass-8 lever 4)")

    # also report the a_term (distogram -> linear_no_bias_a) contribution magnitude
    with open(os.path.join("/home/moritz/.coworker/scratch/odesign-ref/ckpt",
                           "p7_cte_isolation.txt"), "w") as f:
        f.write("cte_isolated_pcc_vs_cpuref=%.6f maxerr=%.4e\n" % (p, m))
        f.write("verdict=%s\n" % ("CLEAN" if p >= 0.999 else "CHECK"))
    print("saved p7_cte_isolation.txt")


if __name__ == "__main__":
    main()
