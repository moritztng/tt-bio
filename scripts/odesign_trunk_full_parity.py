"""Pass-7 on-device FULL-TRUNK parity: run the ttnn ODesignTrunk (48-block Pairformer
+ CTE 2-block pair-only Pairformer + 4-block MSA, 10 recycling cycles) on the
captured prot_binding_trunk inputs, and diff s_inputs / s_trunk / z_trunk against
the CPU-fp32 reference (scripts/odesign_trunk_full_ref.py -- ODesign's own trunk run
fresh in fp32, the rigorous metric) AND against the golden pre (bf16-GPU, the
expected-gap sanity check). Same methodology as passes 1-6: CPU ref = ODesign's own
modules run fresh; bar PCC >= 0.99 for the full trunk.

Run: TT_VISIBLE_DEVICES=0 PYTHONPATH=/home/moritz/.coworker/wt/tt-bio-odesign-port-p8 \
     /home/moritz/tt-bio/env/bin/python3 scripts/odesign_trunk_full_parity.py
"""
import os, sys, pickle, time, torch, ttnn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tt_bio.odesign import ODesignTrunk
from tt_bio.tenstorrent import get_device

CKPT = "/home/moritz/.coworker/scratch/odesign-ref/ckpt/odesign_base_prot_flex.pt"
INP = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_trunk_inputs.pkl"
REF = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_trunk_full_ref.pkl"
PRE = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_denoiser_pre.pkl"
OUT = "/home/moritz/.coworker/scratch/odesign-ref/ckpt/p9_trunk_parity.txt"


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
    trunk = ODesignTrunk(sd, ckc, dev)

    inp = pickle.load(open(INP, "rb"))
    ref = pickle.load(open(REF, "rb"))
    pre = pickle.load(open(PRE, "rb"))
    fd = inp["feature_data"]
    # the captured feature_data has every key ODesignTrunk.__call__ needs (restype,
    # profile, deletion_mean, is_hotspot_residue, token_bonds, msa, has_deletion,
    # deletion_value, constraint_feature, atom feats, token meta).
    feat = {k: fd[k] for k in fd}

    print(f"N_token = {feat['residue_index'].shape[0]}  N_atom = {feat['ref_pos'].shape[0]}")
    n_cycles = int(os.environ.get("ODESIGN_N_CYCLES", "10"))
    per_cycle_dev = []
    def _cap(stage, step=None, total=None, s=None, z=None):
        if stage == "trunk_cycle_end":
            per_cycle_dev.append((step,
                                  trunk._to_host(s, tuple(int(x) for x in s.shape)),
                                  trunk._to_host(z, tuple(int(x) for x in z.shape))))
    t0 = time.time()
    s_trunk_dev, z_trunk_dev = trunk(feat, n_cycles=n_cycles, progress_fn=_cap)
    print(f"on-device trunk run ({n_cycles} cycles, 48 blocks/cycle): {time.time()-t0:.1f}s")
    print(f"s_trunk {tuple(s_trunk_dev.shape)}  z_trunk {tuple(z_trunk_dev.shape)}")

    # per-cycle z/s PCC vs CPU-ref to localize accumulation
    pc = ref.get("per_cycle", [])
    if per_cycle_dev and pc:
        print("\n=== per-cycle drift vs CPU-fp32 reference ===")
        for (cyc, s_h, z_h) in per_cycle_dev:
            if cyc < len(pc):
                ps = pcc(s_h, pc[cyc]["s"].float())
                pz = pcc(z_h, pc[cyc]["z"].float())
                print(f"  cycle {cyc:2d}: s PCC {ps:.6f}  z PCC {pz:.6f}")

    # also re-derive s_inputs from the front-end (pass-6 verified) for the parity table
    s_inputs_dev = trunk.ti.input_feature_embedder(feat)

    results = {}
    # for reduced-cycle smoke tests, compare s_trunk/z_trunk against the matching
    # per-cycle CPU ref (the full ref dumps all 10 cycles).
    pc = ref.get("per_cycle", [])
    cyc_idx = n_cycles - 1
    s_trunk_ref = (pc[cyc_idx]["s"].float() if cyc_idx < len(pc) else ref["s_trunk"].float())
    z_trunk_ref = (pc[cyc_idx]["z"].float() if cyc_idx < len(pc) else ref["z_trunk"].float())
    print("\n=== vs CPU-fp32 reference (rigorous) ===")
    for name, dev_t, ref_t in [("s_inputs", s_inputs_dev, ref["s_inputs"].float()),
                              ("s_trunk", s_trunk_dev, s_trunk_ref),
                              ("z_trunk", z_trunk_dev, z_trunk_ref)]:
        p = pcc(dev_t, ref_t); m = maxerr(dev_t, ref_t)
        print(f"  {name:9s} PCC {p:.6f}  maxerr {m:.4e}")
        results[name + "_vs_cpuref"] = (p, m)

    print("\n=== vs golden pre (bf16-GPU, expected gap; final-cycle only) ===")
    if n_cycles == 10:
        for name, dev_t, gold_t in [("s_inputs", s_inputs_dev, pre["s_inputs"].float()),
                                    ("s_trunk", s_trunk_dev, pre["s_trunk"].float()),
                                    ("z_trunk", z_trunk_dev, pre["z_trunk"].float())]:
            p = pcc(dev_t, gold_t); m = maxerr(dev_t, gold_t)
            print(f"  {name:9s} PCC {p:.6f}  maxerr {m:.4e}")
            results[name + "_vs_golden"] = (p, m)
    else:
        print(f"  (skipped: n_cycles={n_cycles} != 10; golden is the 10-cycle output)")

    # summary
    print("\n=== PASS-7 FULL-TRUNK PARITY SUMMARY ===")
    cp = [v[0] for k, v in results.items() if k.endswith("_vs_cpuref")]
    gp = [v[0] for k, v in results.items() if k.endswith("_vs_golden")]
    print(f"vs CPU-ref:  min PCC {min(cp):.6f}  mean PCC {sum(cp)/len(cp):.6f}")
    if gp:
        print(f"vs golden:   min PCC {min(gp):.6f}  mean PCC {sum(gp)/len(gp):.6f}")
    ok = all(p >= 0.99 for p in cp)
    print("PARITY", "PASS" if ok else "CHECK", "(bar: PCC >= 0.99 vs CPU-ref for s_inputs/s_trunk/z_trunk)")

    # save the on-device trunk outputs for the end-to-end closed-loop test
    if n_cycles == 10:
        DUMP = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_trunk_dev.pkl"
        pickle.dump({"s_inputs": s_inputs_dev, "s_trunk": s_trunk_dev, "z_trunk": z_trunk_dev},
                    open(DUMP, "wb"))
        print("saved", DUMP)

    with open(OUT, "w") as f:
        f.write("=== PASS-7 FULL-TRUNK PARITY ===\n")
        for k, (p, m) in results.items():
            f.write(f"{k}: PCC {p:.6f}  maxerr {m:.4e}\n")
        f.write(f"vs CPU-ref min PCC {min(cp):.6f}  mean {sum(cp)/len(cp):.6f}\n")
        if gp:
            f.write(f"vs golden  min PCC {min(gp):.6f}  mean {sum(gp)/len(gp):.6f}\n")
        f.write("PARITY " + ("PASS" if ok else "CHECK") + "\n")
    print("saved", OUT)


if __name__ == "__main__":
    main()
