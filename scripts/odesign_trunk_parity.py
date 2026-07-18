"""Pass-6 on-device parity: diff the ttnn ODesignTrunkInput port (RPE w/ cyclic,
trunk-init s_init/z_init, ConstraintTemplateEmbedder front-end) against the
CPU-fp32 reference dumped by scripts/odesign_trunk_ref.py (ODesign's own modules
run fresh). Same methodology as passes 1-5: CPU ref = ODesign's own modules.

Run: TT_VISIBLE_DEVICES=0 PYTHONPATH=/home/moritz/.coworker/wt/tt-bio-odesign-port-p6 \
     /home/moritz/tt-bio/env/bin/python3 scripts/odesign_trunk_parity.py
"""
import os, sys, pickle, torch, ttnn
sys.path.insert(0, "/home/moritz/.coworker/wt/tt-bio-odesign-port-p6")
from tt_bio.odesign import ODesignTrunkInput
from tt_bio.tenstorrent import get_device

CKPT = "/home/moritz/.coworker/scratch/odesign-ref/ckpt/odesign_base_prot_flex.pt"
REF = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_trunk_input_ref.pkl"
PRE = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_denoiser_pre.pkl"


def pcc(u, v):
    u = u.flatten().double(); v = v.flatten().double()
    return float(((u - u.mean()) * (v - v.mean())).sum()
                 / ((u - u.mean()).norm() * (v - v.mean()).norm() + 1e-12))


def maxerr(u, v):
    return float((u - v).abs().max())


def load_sd():
    ck = torch.load(CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}


def main():
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = load_sd()
    trunk = ODesignTrunkInput(sd, ckc, dev)
    ref = pickle.load(open(REF, "rb"))
    pre = pickle.load(open(PRE, "rb"))

    results = {}

    # --- (1) RPE non-cyclic (golden pre token meta) ---
    meta = ref["token_meta"]
    relp_dev = trunk.rpe(meta)
    relp_ref = ref["relpe"]
    p = pcc(relp_dev, relp_ref); m = maxerr(relp_dev, relp_ref)
    print(f"RPE (non-cyclic, N={meta['asym_id'].shape[0]})  PCC {p:.6f}  maxerr {m:.4e}")
    results["rpe_noncyclic"] = (p, m)

    # --- (1b) RPE cyclic path (synthetic cyclic peptide, N=12) ---
    cyc = ref["cyclic_meta"]
    relp_cyc_dev = trunk.rpe(cyc)
    relp_cyc_ref = ref["relpe_cyc"]
    p = pcc(relp_cyc_dev, relp_cyc_ref); m = maxerr(relp_cyc_dev, relp_cyc_ref)
    print(f"RPE (cyclic,      N={cyc['asym_id'].shape[0]})  PCC {p:.6f}  maxerr {m:.4e}")
    results["rpe_cyclic"] = (p, m)

    # --- (2) trunk-init s_init / z_init ---
    s_inputs = ref["s_inputs"]
    n = s_inputs.shape[0]
    token_bonds = torch.zeros(n, n)                       # no covalent bonds in this example
    s_init_dev, z_init_dev = trunk.trunk_init(s_inputs, relp_dev, token_bonds)
    p_s = pcc(s_init_dev, ref["s_init"]); m_s = maxerr(s_init_dev, ref["s_init"])
    p_z = pcc(z_init_dev, ref["z_init"]); m_z = maxerr(z_init_dev, ref["z_init"])
    print(f"trunk s_init (N={n},384)  PCC {p_s:.6f}  maxerr {m_s:.4e}")
    print(f"trunk z_init (N={n},{n},128) PCC {p_z:.6f}  maxerr {m_z:.4e}")
    results["s_init"] = (p_s, m_s); results["z_init"] = (p_z, m_z)

    # --- (3) ConstraintTemplateEmbedder front-end (v_ij) ---
    cf = ref["constraint_feature"]
    v_dev = trunk.constraint_embedder_front(z_init_dev, cf)
    v_ref = ref["v_ij"]
    p = pcc(v_dev, v_ref); m = maxerr(v_dev, v_ref)
    print(f"CTE front v_ij (N={n},{n},64)  PCC {p:.6f}  maxerr {m:.4e}")
    results["cte_front"] = (p, m)

    # --- (4) InputFeatureEmbedder (atom encoder -> s_inputs) ---
    # uses the captured trunk inputs (scripts/odesign_trunk_input_capture.py) +
    # the CPU baseline (scripts/odesign_s_inputs_ref.py)
    INP = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_trunk_inputs.pkl"
    SREF = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_s_inputs_ref.pkl"
    if os.path.exists(INP) and os.path.exists(SREF):
        inp = pickle.load(open(INP, "rb"))
        sref = pickle.load(open(SREF, "rb"))
        fd = inp["feature_data"]
        s_dev = trunk.input_feature_embedder(fd)
        s_golden = pre["s_inputs"].float()                       # the golden pre's s_inputs
        s_cpu = sref["s_inputs_ref"].float()                     # ODesign's own, fresh CPU fp32
        p_g = pcc(s_dev, s_golden); m_g = maxerr(s_dev, s_golden)
        p_c = pcc(s_dev, s_cpu); m_c = maxerr(s_dev, s_cpu)
        print(f"InputFeatureEmbedder s_inputs (N={s_dev.shape[0]},453)  vs golden  PCC {p_g:.6f}  maxerr {m_g:.4e}")
        print(f"InputFeatureEmbedder s_inputs (N={s_dev.shape[0]},453)  vs CPU-ref PCC {p_c:.6f}  maxerr {m_c:.4e}")
        results["s_inputs_vs_golden"] = (p_g, m_g)
        results["s_inputs_vs_cpuref"] = (p_c, m_c)
    else:
        print("InputFeatureEmbedder parity SKIPPED (trunk-input capture / CPU baseline not run)")

    # summary
    print("\n=== PASS-6 TRUNK-INPUT PARITY SUMMARY ===")
    allp = [v[0] for v in results.values()]
    print(f"min PCC {min(allp):.6f}  mean PCC {sum(allp)/len(allp):.6f}  across {len(allp)} components")
    ok = all(p >= 0.999 for p, _ in results.values())
    print("PARITY", "PASS" if ok else "CHECK", "(bar: PCC >= 0.999 for every component)")


if __name__ == "__main__":
    main()
