# Undersampling ablation for on-device Protenix-v2 (tt-bio OWN device impl).
# Runs the SAME input (275-atom / 38-token molecule assembled from the golden pkls -- the
# same feats the e2e capstone used) across a sweep of diffusion sampling settings and reports
# Kabsch-aligned RMSD, seed-to-seed spread, and best-of-N pLDDT vs wall-clock cost.
#
# EFFICIENCY: the 10-cycle trunk + conditioning are t-independent and are computed ONCE; only
# edm_sample (the diffusion denoise loop) is swept over n_step / seed. Conditioning is
# replicated verbatim from Protenix.fold (steps 1-4) so the numbers match the fold() path.
#
# Reference target = protenix_traj.pkl final_coords: the OFFICIAL Protenix-v2 reference model
# prediction on this exact molecule (captured at its config N_step=10). This is a reference-
# PREDICTION, not an experimental ground-truth structure (none is available on this host); the
# report is explicit about that. Env knobs: N_STEPS (csv), SEEDS (csv), PROBE=1 (tiny probe).
import os, sys, time, pickle
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tt-bio repo root
os.environ.setdefault("TT_VISIBLE_DEVICES", "3")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
sys.path.insert(0, _REPO)
import torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Protenix, edm_sample
import tt_bio.tenstorrent as _TT

CKPT = "/home/ttuser/protenix_ckpt/protenix-v2.pt"
ife = pickle.load(open("/home/ttuser/protenix_ife_gold.pkl", "rb"))
tg  = pickle.load(open("/home/ttuser/protenix_trunkin_gold.pkl", "rb"))
d   = pickle.load(open("/home/ttuser/protenix_ref_out.pkl", "rb"))
tfeat = d["intermediates"]["template_embedder"]["in"][0]
cp  = pickle.load(open("/home/ttuser/protenix_confidence_pre.pkl", "rb"))["kwargs"]["input_feature_dict"]
traj = pickle.load(open("/home/ttuser/protenix_traj.pkl", "rb"))

F = ife["feat"]
feats = {
    "ref_pos": F["ref_pos"], "ref_charge": F["ref_charge"], "ref_mask": F["ref_mask"],
    "ref_element": F["ref_element"], "ref_atom_name_chars": F["ref_atom_name_chars"],
    "d_lm": F["d_lm"], "v_lm": F["v_lm"], "atom_to_token_idx": F["atom_to_token_idx"],
    "restype": F["restype"], "profile": F["profile"], "deletion_mean": F["deletion_mean"],
    "mask_trunked": ife["mask_trunked"],
    "relp": tg["relp"], "token_bonds": tg["token_bonds"],
    "template_aatype": tfeat["template_aatype"], "template_distogram": tfeat["template_distogram"],
    "template_pseudo_beta_mask": tfeat["template_pseudo_beta_mask"],
    "template_unit_vector": tfeat["template_unit_vector"],
    "template_backbone_frame_mask": tfeat["template_backbone_frame_mask"],
    "msa": tfeat["msa"], "has_deletion": tfeat["has_deletion"], "deletion_value": tfeat["deletion_value"],
    "asym_id": tfeat["asym_id"],
}
# extra feats needed only by the confidence head (best-of-N ranking), same molecule
for k in ("distogram_rep_atom_mask", "atom_to_tokatom_idx"):
    feats[k] = cp[k]

def _procrustes(P, Q, allow_reflection):
    # allow_reflection=False -> proper-rotation Kabsch; True -> optimal orthogonal transform
    # (rotation OR reflection), so a mirror-image structure collapses to a small RMSD. The
    # gap between the two is a direct chirality/handedness-flip test.
    Pc = P - P.mean(0); Qc = Q - Q.mean(0); H = Pc.t() @ Qc
    U, _, Vt = torch.linalg.svd(H)
    if allow_reflection:
        R = Vt.t() @ U.t()
    else:
        Dm = torch.diag(torch.tensor([1., 1., torch.sign(torch.det(Vt.t() @ U.t()))]))
        R = Vt.t() @ Dm @ U.t()
    return float(((Pc @ R.t()) - Qc).pow(2).sum(-1).mean().sqrt())

def kabsch(P, Q):
    return _procrustes(P, Q, allow_reflection=False)

def kabsch_mirror(P, Q):
    return _procrustes(P, Q, allow_reflection=True)

def rg(x):
    return float((x - x.mean(0)).pow(2).sum(-1).mean().sqrt())

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
model = Protenix.load_from_checkpoint(CKPT, compute_kernel_config=ckc, device=dev)

# ---- conditioning (once) : verbatim from Protenix.fold steps 1-4 ----
t0 = time.time()
fi = model._atom_feat_inputs(feats)
N, NT, nb, nq, nk = fi["N"], fi["NT"], fi["nb"], fi["nq"], fi["nk"]
mt = fi["mt"]; S = fi["S"]; tt = model._tt
Mmat = (S.t() / (S.t().sum(-1, keepdim=True) + 1e-6))
dm = feats["deletion_mean"]; dm = dm.reshape(-1, 1) if dm.dim() == 1 else dm
s_inputs_tt = model.input_aae(tt(feats["ref_pos"]), tt(fi["ref_charge_asinh"]), tt(feats["ref_mask"].reshape(N, 1)),
                              tt(fi["f_in"]), tt(fi["d"]), tt(fi["v"]), tt(fi["invd"]), mt, tt(Mmat),
                              tt(feats["restype"]), tt(feats["profile"]), tt(dm))
s_inputs = model._to_host(s_inputs_tt)[:NT]
mt_dev = tt(mt.reshape(-1, 1).float())
c_l = model._to_host(model.diff_feat.c_l(tt(feats["ref_pos"]), tt(fi["ref_charge_asinh"]),
                                         tt(feats["ref_mask"].reshape(N, 1)), tt(fi["f_in"])), (N, 128))
p_lm = model._to_host(model.diff_feat.p_lm(tt(fi["d"]), tt(fi["v"]), tt(fi["invd"]), mt_dev), (nb, nq, nk, 16))
relp = feats["relp"] if "relp" in feats else model._generate_relp(feats)
if model._fast: _TT.set_fast_mode(True)
s_trunk_tt, z_tt = model.trunk(feats, s_inputs, relp, feats["token_bonds"], progress_fn=None, n_cycles=None)
if model._fast: _TT.set_fast_mode(False)
s_trunk = model._to_host(s_trunk_tt, (NT, s_trunk_tt.shape[-1]))
z_trunk = model._to_host(z_tt, (NT, NT, model.trunk.C_Z))
pair_z = model._diffusion_pair_cond(z_tt, relp).reshape(NT, NT, model.trunk.C_Z)
p_lm = p_lm + model._plm_z_term(pair_z, fi["a2t"], nb, nq, nk)
cond = {"s_trunk": s_trunk, "s_inputs": s_inputs, "pair_z": pair_z, "c_l": c_l,
        "p_lm": p_lm, "S": S, "mask_trunked": mt.float()}
if model.diffusion.device_dit:
    cond["dit_z"] = model.diffusion._dit_z_device(pair_z)
else:
    cond["dit_biases"] = model.diffusion._dit_pair_biases(pair_z)
t_trunk = time.time() - t0
print("[cond] trunk+conditioning once: %.1fs  N_atoms=%d NT=%d" % (t_trunk, N, NT), flush=True)

ref = traj["final_coords"].float().reshape(-1, 3)[:N]
print("[ref] official Protenix N_step=10 pred: %d atoms  Rg=%.2fA" % (ref.shape[0], rg(ref)), flush=True)

PROBE = os.environ.get("PROBE") == "1"
N_STEPS = [int(x) for x in os.environ.get("N_STEPS", "10,25,50,100,200").split(",")]
SEEDS   = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
if PROBE:
    N_STEPS, SEEDS = [10], [0]

print("\n n_step | seeds | RMSD_vs_ref(best-pLDDT) | RMSD_vs_ref(mean) | seed2seed | best_pLDDT | edm_s | Rg", flush=True)
for n_step in N_STEPS:
    ts = time.time()
    samples = []
    for sd in SEEDS:
        samples.append(edm_sample(model.diffusion, cond, N, n_step=n_step, seed=sd)[0])
    dt = time.time() - ts
    confs = [model.confidence_head.confidence(s_inputs, s_trunk, z_trunk, s, feats) for s in samples]
    plddts = [c["plddt"] for c in confs]
    best = int(torch.tensor(plddts).argmax())
    rmsd_ref = [kabsch(s, ref) for s in samples]
    rmsd_ref_m = [kabsch_mirror(s, ref) for s in samples]
    s2s = kabsch(samples[0], samples[1]) if len(samples) > 1 else float("nan")
    print(" %6d | %5d | %22.3f | %17.3f | %9.3f | %10.4f | %5.1f | %.2f" % (
        n_step, len(SEEDS), rmsd_ref[best], sum(rmsd_ref) / len(rmsd_ref), s2s,
        plddts[best], dt / len(SEEDS), rg(samples[best])), flush=True)
    print("        per-seed RMSD_vs_ref =%s  pLDDT=%s" % (
        ["%.2f" % r for r in rmsd_ref], ["%.3f" % p for p in plddts]), flush=True)
    print("        per-seed MIRROR_vs_ref=%s  (proper-mirror gap=%s)" % (
        ["%.2f" % r for r in rmsd_ref_m],
        ["%.2f" % (rmsd_ref[i] - rmsd_ref_m[i]) for i in range(len(samples))]), flush=True)
print("ABLATION_DONE", flush=True)
