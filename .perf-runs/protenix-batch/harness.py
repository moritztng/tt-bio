import os, sys, time, pickle, argparse
os.environ.setdefault('TT_LOGGER_LEVEL', 'FATAL')
# use THIS repo's tt_bio (not the legacy /home/ttuser/tt-boltz2)
sys.path.insert(0, '/home/ttuser/tt-bio-dev')
import torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Protenix

ap = argparse.ArgumentParser()
ap.add_argument('--n_step', type=int, default=60)
ap.add_argument('--n_sample', type=int, default=2)
ap.add_argument('--fast', action='store_true')
args = ap.parse_args()

CKPT = '/home/ttuser/protenix_ckpt/protenix-v2.pt'
ife = pickle.load(open('/home/ttuser/protenix_ife_gold.pkl', 'rb'))
tg = pickle.load(open('/home/ttuser/protenix_trunkin_gold.pkl', 'rb'))
d = pickle.load(open('/home/ttuser/protenix_ref_out.pkl', 'rb'))
tfeat = d['intermediates']['template_embedder']['in'][0]
traj = pickle.load(open('/home/ttuser/protenix_traj.pkl', 'rb'))
F = ife['feat']
feats = {
    'ref_pos': F['ref_pos'], 'ref_charge': F['ref_charge'], 'ref_mask': F['ref_mask'],
    'ref_element': F['ref_element'], 'ref_atom_name_chars': F['ref_atom_name_chars'],
    'd_lm': F['d_lm'], 'v_lm': F['v_lm'], 'atom_to_token_idx': F['atom_to_token_idx'],
    'restype': F['restype'], 'profile': F['profile'], 'deletion_mean': F['deletion_mean'],
    'mask_trunked': ife['mask_trunked'],
    'relp': tg['relp'], 'token_bonds': tg['token_bonds'],
    'template_aatype': tfeat['template_aatype'], 'template_distogram': tfeat['template_distogram'],
    'template_pseudo_beta_mask': tfeat['template_pseudo_beta_mask'],
    'template_unit_vector': tfeat['template_unit_vector'],
    'template_backbone_frame_mask': tfeat['template_backbone_frame_mask'],
    'msa': tfeat['msa'], 'has_deletion': tfeat['has_deletion'], 'deletion_value': tfeat['deletion_value'],
    'asym_id': tfeat['asym_id'],
}
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
model = Protenix.load_from_checkpoint(CKPT, compute_kernel_config=ckc, device=dev)
model._fast = bool(args.fast)
print(f"N_atoms={F['ref_pos'].shape[0]} NT={F['restype'].shape[0]} fast={args.fast} "
      f"n_step={args.n_step} n_sample={args.n_sample}", flush=True)
t0 = time.time()
coords = model.fold(feats, n_step=args.n_step, n_sample=args.n_sample, seed=0)
print(f"fold total {time.time()-t0:.3f}s  coords={tuple(coords.shape)} finite={bool(torch.isfinite(coords).all())}", flush=True)

def kabsch(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0); H = Pc.t() @ Qc
    U, _, Vt = torch.linalg.svd(H)
    Dm = torch.diag(torch.tensor([1., 1., torch.sign(torch.det(Vt.t() @ U.t()))]))
    R = Vt.t() @ Dm @ U.t()
    return float(((Pc @ R.t()) - Qc).pow(2).sum(-1).mean().sqrt())

x0 = coords[0]
print('Rg %.3f A' % float((x0 - x0.mean(0)).pow(2).sum(-1).mean().sqrt()), flush=True)
for k in range(1, coords.shape[0]):
    print('seed0-vs-seed%d Kabsch RMSD: %.4f A' % (k, kabsch(x0, coords[k])), flush=True)
ref = traj.get('final_coords')
if ref is not None:
    rf = ref.float().reshape(-1, 3)[:x0.shape[0]]
    print('seed0 vs reference Kabsch RMSD: %.4f A' % kabsch(x0, rf), flush=True)
print('HARNESS_DONE', flush=True)
